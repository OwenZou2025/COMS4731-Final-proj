// Wardrobe Management and UI

import { state, MAX_ITEMS } from './state.js';
import { toast } from './toast.js';
import { getErrorHint } from './utils.js';
import { 
    getAllWardrobeItems, 
    deleteWardrobeItem, 
    saveWardrobeOrder,
    getDb
} from './db.js';
import { getRecommendation, showLoading, hideLoading } from './analysis.js';
import { getWardrobeRecommendationML } from './ml-wardrobe.js';
import { switchView } from './navigation.js';

// Drag and drop state
let draggedItem = null;
let draggedItemId = null;

// Delete modal state
let itemToDelete = null;
let deleteMultipleItems = false;

// Context listeners setup flag
let wardrobeListenersSetup = false;

export function initWardrobe() {
    // Clear wardrobe selections button
    const clearWardrobeBtn = document.getElementById('clearWardrobe');
    if (clearWardrobeBtn) {
        clearWardrobeBtn.addEventListener('click', () => {
            // Clear selections only, don't delete items
            toast.dismissAll(); // Dismiss any limit warnings
            state.selectedWardrobeItems = [];
            state.usedSelectAll = false; // Reset Select All flag
            loadWardrobe();
            updateSelectedWardrobeDisplay();
            updateWardrobeSubmitButton();
        });
    }

    // Select All wardrobe items
    const selectAllWardrobeBtn = document.getElementById('selectAllWardrobe');
    if (selectAllWardrobeBtn) {
        selectAllWardrobeBtn.addEventListener('click', async () => {
            const allItems = await getAllWardrobeItems();
            // Select ALL items (no limit)
            state.selectedWardrobeItems = allItems.map(item => item.id);
            state.usedSelectAll = true; // Mark that Select All was used
            loadWardrobe();
            updateSelectedWardrobeDisplay();
            updateWardrobeSubmitButton();
        });
    }

    // Get recommendation from wardrobe button
    const getWardrobeRecommendationBtn = document.getElementById('getWardrobeRecommendation');
    if (getWardrobeRecommendationBtn) {
        getWardrobeRecommendationBtn.addEventListener('click', async () => {
            await getWardrobeRecommendation();
        });
    }

    // Modal event listeners
    const cancelDeleteBtn = document.getElementById('cancelDeleteWardrobe');
    if (cancelDeleteBtn) {
        cancelDeleteBtn.addEventListener('click', () => {
            hideDeleteWardrobeModal();
        });
    }

    const confirmDeleteBtn = document.getElementById('confirmDeleteWardrobe');
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', async () => {
            if (itemToDelete) {
                try {
                    if (deleteMultipleItems && Array.isArray(itemToDelete)) {
                        // Delete multiple items
                        for (const item of itemToDelete) {
                            await deleteWardrobeItem(item.id);
                        }
                        state.selectedWardrobeItems = [];
                    } else {
                        // Delete single item
                        await deleteWardrobeItem(itemToDelete.id);
                        // Remove from selection if it was selected
                        state.selectedWardrobeItems = state.selectedWardrobeItems.filter(itemId => itemId !== itemToDelete.id);
                    }
                    loadWardrobe();
                    updateSelectedWardrobeDisplay();
                    hideDeleteWardrobeModal();
                } catch (error) {
                    console.error('Error deleting wardrobe item:', error);
                    const hint = getErrorHint(error);
                    toast.error('Delete failed', `Couldn't remove from wardrobe.${hint ? ' ' + hint : ''}`);
                }
            }
        });
    }

    // Close modal when clicking outside
    const deleteModal = document.getElementById('deleteWardrobeModal');
    if (deleteModal) {
        deleteModal.addEventListener('click', (e) => {
            if (e.target.id === 'deleteWardrobeModal') {
                hideDeleteWardrobeModal();
            }
        });
    }
}

export async function loadWardrobe() {
    const wardrobeGrid = document.getElementById('wardrobeGrid');
    const clearSelectionBtn = document.getElementById('clearWardrobe');
    const selectAllBtn = document.getElementById('selectAllWardrobe');
    
    // Hide clear selection button initially if nothing is selected
    if (clearSelectionBtn && state.selectedWardrobeItems.length === 0) {
        clearSelectionBtn.style.display = 'none';
    }
    
    try {
        const items = await getAllWardrobeItems();
        
        // Show/hide Select All button based on wardrobe contents
        if (selectAllBtn) {
            if (items.length === 0) {
                selectAllBtn.style.display = 'none';
            } else {
                selectAllBtn.style.display = 'flex';
            }
        }
        
        if (items.length === 0) {
            wardrobeGrid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="12" y1="8" x2="12" y2="16"></line>
                        <line x1="8" y1="12" x2="16" y2="12"></line>
                    </svg>
                    <h3>Your wardrobe is empty</h3>
                    <p>Save items from the upload page to build your wardrobe</p>
                </div>
            `;
            return;
        }
        
        wardrobeGrid.innerHTML = '';
        
        // Sort by order field (if exists) then by timestamp
        items.sort((a, b) => {
            // If both have order, sort by order (ascending)
            if (a.order !== undefined && b.order !== undefined) {
                return a.order - b.order;
            }
            // Items with order come before items without
            if (a.order !== undefined) return -1;
            if (b.order !== undefined) return 1;
            // Fallback to timestamp (newest first)
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
        
        items.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = 'wardrobe-item';
            div.draggable = true;
            div.dataset.id = item.id;
            div.dataset.index = index;
            
            if (state.selectedWardrobeItems.includes(item.id)) {
                div.classList.add('selected-for-styling');
            }
            
            const date = new Date(item.timestamp);
            const dateStr = date.toLocaleDateString();
            
            div.innerHTML = `
                <div class="wardrobe-image">
                    <img src="${item.imageData}" alt="${item.fileName}" draggable="false">
                    <button class="delete-wardrobe-item" data-id="${item.id}" onclick="event.stopPropagation();">&times;</button>
                    ${state.selectedWardrobeItems.includes(item.id) ? '<div class="selection-badge">✓ Selected</div>' : ''}
                </div>
                <div class="wardrobe-label">
                    <div class="wardrobe-date">${dateStr}</div>
                </div>
            `;
            
            // Click to select/deselect for styling
            div.addEventListener('click', (e) => {
                if (!e.target.classList.contains('delete-wardrobe-item') && !draggedItem) {
                    toggleWardrobeItemSelection(item.id);
                }
            });
            
            // Drag and drop events
            div.addEventListener('dragstart', handleDragStart);
            div.addEventListener('dragend', handleDragEnd);
            div.addEventListener('dragover', handleDragOver);
            div.addEventListener('dragenter', handleDragEnter);
            div.addEventListener('dragleave', handleDragLeave);
            div.addEventListener('drop', handleDrop);
            
            wardrobeGrid.appendChild(div);
        });
        
        // Add delete event listeners
        wardrobeGrid.querySelectorAll('.delete-wardrobe-item').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const id = parseInt(btn.dataset.id);
                const item = items.find(i => i.id === id);
                if (item) {
                    showDeleteWardrobeModal(item);
                }
            });
        });
        
    } catch (error) {
        console.error('Error loading wardrobe:', error);
        wardrobeGrid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <h3>Error loading wardrobe</h3>
                <p>Please refresh the page and try again</p>
            </div>
        `;
    }
}

// Drag and Drop handlers
function handleDragStart(e) {
    draggedItem = this;
    draggedItemId = parseInt(this.dataset.id);
    
    this.classList.add('dragging');
    const wardrobeGrid = document.getElementById('wardrobeGrid');
    wardrobeGrid.classList.add('drag-active');
    
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', this.dataset.id);
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    const wardrobeGrid = document.getElementById('wardrobeGrid');
    wardrobeGrid.classList.remove('drag-active');
    
    // Remove drag-over class from all items
    document.querySelectorAll('.wardrobe-item.drag-over').forEach(item => {
        item.classList.remove('drag-over');
    });
    
    draggedItem = null;
    draggedItemId = null;
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
}

function handleDragEnter(e) {
    e.preventDefault();
    if (this !== draggedItem) {
        this.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    this.classList.remove('drag-over');
}

async function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    this.classList.remove('drag-over');
    
    if (this === draggedItem) return;
    
    const dropTargetId = parseInt(this.dataset.id);
    const draggedId = draggedItemId;
    
    if (dropTargetId === draggedId) return;
    
    // Get current order of items
    const wardrobeGrid = document.getElementById('wardrobeGrid');
    const items = Array.from(wardrobeGrid.querySelectorAll('.wardrobe-item'));
    const itemIds = items.map(item => parseInt(item.dataset.id));
    
    // Find positions
    const draggedIndex = itemIds.indexOf(draggedId);
    const dropIndex = itemIds.indexOf(dropTargetId);
    
    if (draggedIndex === -1 || dropIndex === -1) return;
    
    // Reorder the array
    itemIds.splice(draggedIndex, 1);
    itemIds.splice(dropIndex, 0, draggedId);
    
    // Save new order to IndexedDB
    await saveWardrobeOrder(itemIds);
    
    // Reload wardrobe to reflect new order
    await loadWardrobe();
    updateSelectedWardrobeDisplay();
}

function toggleWardrobeItemSelection(itemId) {
    const index = state.selectedWardrobeItems.indexOf(itemId);
    
    if (index === -1) {
        // Check limit
        if (state.selectedWardrobeItems.length >= MAX_ITEMS) {
            toast.warning('Selection limit reached', `You can style up to ${MAX_ITEMS} items at once for the best results.`);
            return;
        }
        state.selectedWardrobeItems.push(itemId);
    } else {
        state.selectedWardrobeItems.splice(index, 1);
        toast.dismissAll(); // Dismiss any warnings when unselecting
    }
    
    // Only reset Select All flag when count goes back to ≤10
    if (state.selectedWardrobeItems.length <= MAX_ITEMS) {
        state.usedSelectAll = false;
    }
    
    loadWardrobe();
    updateSelectedWardrobeDisplay();
}

export async function updateSelectedWardrobeDisplay() {
    const selectedSection = document.getElementById('selectedWardrobeSection');
    const selectedItemsDiv = document.getElementById('selectedWardrobeItems');
    const clearSelectionBtn = document.getElementById('clearWardrobe');
    
    if (state.selectedWardrobeItems.length === 0) {
        selectedSection.classList.add('hidden');
        if (clearSelectionBtn) {
            clearSelectionBtn.style.display = 'none';
        }
        return;
    }
    
    selectedSection.classList.remove('hidden');
    if (clearSelectionBtn) {
        clearSelectionBtn.style.display = 'flex';
    }
    
    // Update selection counter
    const counterEl = document.getElementById('wardrobeSelectionCounter');
    const count = state.selectedWardrobeItems.length;
    if (counterEl) {
        if (count > MAX_ITEMS) {
            counterEl.textContent = `(${count} selected)`;
        } else {
            counterEl.textContent = `(${count}/${MAX_ITEMS})`;
        }
    }
    
    // Show/hide styling options based on Select All usage
    const wardrobeContextSection = document.getElementById('wardrobeContextSection');
    const wardrobeRecommendationBtn = document.getElementById('getWardrobeRecommendation');
    const hideStylingOptions = state.usedSelectAll && count > MAX_ITEMS;
    
    if (wardrobeContextSection) {
        wardrobeContextSection.style.display = hideStylingOptions ? 'none' : '';
    }
    if (wardrobeRecommendationBtn) {
        wardrobeRecommendationBtn.style.display = hideStylingOptions ? 'none' : '';
    }
    
    // Remove existing header if present
    const existingHeader = selectedSection.querySelector('.selected-wardrobe-header');
    if (existingHeader) {
        existingHeader.remove();
    }
    
    selectedItemsDiv.innerHTML = '';
    
    // Add header with delete button
    const header = document.createElement('div');
    header.className = 'selected-wardrobe-header';
    header.innerHTML = `
        <h3>Selected Items (${state.selectedWardrobeItems.length})</h3>
        <button class="btn-delete-selected" id="deleteSelectedItems">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                <line x1="10" y1="11" x2="10" y2="17"></line>
                <line x1="14" y1="11" x2="14" y2="17"></line>
            </svg>
            Delete Selected
        </button>
    `;
    selectedSection.insertBefore(header, selectedItemsDiv);
    
    // Add delete handler
    const deleteBtn = document.getElementById('deleteSelectedItems');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', deleteSelectedWardrobeItems);
    }
    
    try {
        const allItems = await getAllWardrobeItems();
        // Preserve selection order by mapping from selectedWardrobeItems
        const selectedItems = state.selectedWardrobeItems
            .map(id => allItems.find(item => item.id === id))
            .filter(item => item); // Remove any undefined
        
        selectedItems.forEach(item => {
            const div = document.createElement('div');
            div.className = 'selected-wardrobe-preview';
            div.innerHTML = `
                <img src="${item.imageData}" alt="${item.fileName}">
            `;
            selectedItemsDiv.appendChild(div);
        });
        
    } catch (error) {
        console.error('Error updating selected wardrobe display:', error);
    }
    
    updateWardrobeSubmitButton();
}

// Wardrobe context selection - setup event listeners
export function setupWardrobeContextListeners() {
    if (wardrobeListenersSetup) return; // Prevent duplicate listeners
    
    const wardrobeContextButtons = document.querySelectorAll('.wardrobe-context-btn');
    const wardrobeCustomContextInput = document.getElementById('wardrobeCustomContext');
    
    if (!wardrobeContextButtons.length || !wardrobeCustomContextInput) return;

    wardrobeContextButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            wardrobeContextButtons.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            state.wardrobeContext = btn.dataset.context;
            wardrobeCustomContextInput.value = '';
            state.wardrobeCustomContext = '';
            updateWardrobeSubmitButton();
        });
    });

    wardrobeCustomContextInput.addEventListener('input', (e) => {
        state.wardrobeCustomContext = e.target.value;
        if (state.wardrobeCustomContext) {
            wardrobeContextButtons.forEach(b => b.classList.remove('selected'));
            state.wardrobeContext = null;
        }
        updateWardrobeSubmitButton();
    });
    
    // Deselect occasion buttons when user clicks on custom input
    wardrobeCustomContextInput.addEventListener('focus', () => {
        wardrobeContextButtons.forEach(b => b.classList.remove('selected'));
        state.wardrobeContext = null;
        updateWardrobeSubmitButton();
    });
    
    wardrobeListenersSetup = true;
}

export function updateWardrobeSubmitButton() {
    const btn = document.getElementById('getWardrobeRecommendation');
    if (!btn) return;
    
    const hasItems = state.selectedWardrobeItems.length > 0;
    const hasContext = state.wardrobeContext || state.wardrobeCustomContext;
    btn.disabled = !(hasItems && hasContext);
}

async function getWardrobeRecommendation() {
    try {
        // Get selected wardrobe items (preserve selection order)
        const allItems = await getAllWardrobeItems();
        const selectedItems = state.selectedWardrobeItems
            .map(id => allItems.find(item => item.id === id))
            .filter(item => item); // Remove any undefined
        
        // Early return if no items (button should be disabled, but just in case)
        if (selectedItems.length === 0) return;
        
        // Early return if too many items (button should be hidden, but just in case)
        if (selectedItems.length > MAX_ITEMS) return;
        
        // Show loading state immediately (no need to switch views - loading overlays all views)
        showLoading();
        
        // If using ML backend, use special wardrobe analysis pathway
        if (state.useMLBackend) {
            console.log('Using ML backend for wardrobe analysis...');
            const { displayResults } = await import('./results.js');
            await getWardrobeRecommendationML(displayResults);
            hideLoading();
            return;
        }
        
        // Convert wardrobe items to files for the API
        const files = [];
        const imageURLs = [];
        
        for (let i = 0; i < selectedItems.length; i++) {
            const item = selectedItems[i];
            // Convert data URL to blob
            const response = await fetch(item.imageData);
            const blob = await response.blob();
            const file = new File([blob], item.fileName, { type: blob.type });
            files.push(file);
            imageURLs.push(item.imageData);
        }
        
        // Temporarily set state for the recommendation process
        const originalFiles = state.uploadedFiles;
        const originalImageURLs = state.uploadedImageURLs;
        
        state.uploadedFiles = files;
        state.uploadedImageURLs = imageURLs;
        
        // Use wardrobe context
        const originalContext = state.customContext;
        const originalSelectedContext = state.selectedContext;
        
        state.customContext = state.wardrobeCustomContext;
        state.selectedContext = state.wardrobeContext;
        
        // Set analysis source to wardrobe
        state.analysisSource = 'wardrobe';
        
        // Get recommendation
        await getRecommendation();
        
        // Restore original state
        state.uploadedFiles = originalFiles;
        state.uploadedImageURLs = originalImageURLs;
        state.customContext = originalContext;
        state.selectedContext = originalSelectedContext;
        
    } catch (error) {
        console.error('Error getting wardrobe recommendation:', error);
        // Show error toast
        const hint = getErrorHint(error);
        toast.error('Analysis failed', `Couldn't get a recommendation.${hint ? ' ' + hint : ''}`);
        // Hide loading state and switch view
        hideLoading();
        switchView('wardrobe', false);
    }
}

// Delete Wardrobe Modal Functions
function showDeleteWardrobeModal(item) {
    toast.dismissAll(); // Dismiss any limit warnings
    itemToDelete = item;
    deleteMultipleItems = false;
    const modal = document.getElementById('deleteWardrobeModal');
    const title = modal.querySelector('.modal-title');
    const description = modal.querySelector('.modal-description');
    
    if (Array.isArray(item)) {
        deleteMultipleItems = true;
        title.textContent = `Remove ${item.length} items from your wardrobe?`;
        description.textContent = 'This action cannot be undone.';
    } else {
        title.textContent = 'Remove this item from your wardrobe?';
        description.textContent = 'This action cannot be undone.';
    }
    
    modal.classList.remove('hidden');
}

function hideDeleteWardrobeModal() {
    const modal = document.getElementById('deleteWardrobeModal');
    modal.classList.add('hidden');
    itemToDelete = null;
    deleteMultipleItems = false;
}

async function deleteSelectedWardrobeItems() {
    if (state.selectedWardrobeItems.length === 0) return;
    
    const items = await getAllWardrobeItems();
    const itemsToDelete = items.filter(item => state.selectedWardrobeItems.includes(item.id));
    
    showDeleteWardrobeModal(itemsToDelete);
}
