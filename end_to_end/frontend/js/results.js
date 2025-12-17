// Results Display and Formatting

import { state } from './state.js';
import { toTitleCase, formatItemName, markdownToHTML, getErrorHint, compressImage } from './utils.js';
import { toast } from './toast.js';
import { switchView } from './navigation.js';
import { addToHistoryDB, getHistoryItemDB, addToWardrobe } from './db.js';
import { resetUploadForm } from './upload.js';
import { hideLoading } from './analysis.js';

// DOM Elements
let loadingState;

export function initResults() {
    loadingState = document.getElementById('loadingState');

    // Back/Edit button handler
    const backBtn = document.getElementById('backToUpload');
    if (backBtn) {
        backBtn.addEventListener('click', () => {
            sessionStorage.removeItem('currentResult');
            sessionStorage.removeItem('lastViewedResultTimestamp');
            
            const source = state.analysisSource;
            
            if (source === 'history') {
                // Go back to history page
                state.analysisSource = 'upload'; // Reset for next time
                switchView('history', true);
            } else if (source === 'wardrobe') {
                // Go back to wardrobe with items still selected but clear occasion for new entry
                state.wardrobeContext = null;
                state.wardrobeCustomContext = '';
                
                // switchView will preserve items because analysisSource is still 'wardrobe'
                switchView('wardrobe', true);
                
                // Now reset analysisSource after switch is complete
                state.analysisSource = 'upload';
                
                // Clear occasion inputs
                const wardrobeCustomContextInput = document.getElementById('wardrobeCustomContext');
                if (wardrobeCustomContextInput) {
                    wardrobeCustomContextInput.value = '';
                }
                const wardrobeContextButtons = document.querySelectorAll('.wardrobe-context-btn');
                wardrobeContextButtons.forEach(btn => btn.classList.remove('selected'));
                
                // updateWardrobeSubmitButton will be called via the callback
            } else {
                // Default: go to upload
                state.isFromHistory = false;
                state.analysisSource = 'upload';
                switchView('upload', true);
                resetUploadForm();
            }
        });
    }

    // Save to History button handler
    const saveToHistoryBtn = document.getElementById('saveToHistory');
    if (saveToHistoryBtn) {
        saveToHistoryBtn.addEventListener('click', async () => {
            if (!state.currentResult) return;

            const saveBtn = document.getElementById('saveToHistory');
            const originalText = saveBtn.textContent;
            
            // Show saving state
            saveBtn.textContent = 'Saving...';
            saveBtn.disabled = true;
            
            // Ensure it's saved (will only save if not already in history)
            const wasSaved = await saveResultToHistory(state.currentResult);
            
            if (wasSaved) {
                // Mark as from history and clear the temporary storage
                state.isFromHistory = true;
                sessionStorage.removeItem('currentResult');
                sessionStorage.setItem('lastViewedResultTimestamp', state.currentResult.timestamp);
                
                // Update button text to show confirmation
                saveBtn.textContent = 'Saved ✓';
                saveBtn.style.backgroundColor = '#10b981';
                saveBtn.style.borderColor = '#10b981';
            } else {
                // Already saved - just update button
                saveBtn.textContent = 'Already Saved';
                saveBtn.style.backgroundColor = 'var(--text-secondary)';
                saveBtn.style.borderColor = 'var(--text-secondary)';
            }
            
            saveBtn.disabled = false;
            
            // Reset button after 2 seconds
            setTimeout(() => {
                saveBtn.textContent = originalText;
                saveBtn.style.backgroundColor = '';
                saveBtn.style.borderColor = '';
            }, 2000);
        });
    }
    
    // Event delegation for save-mask-btn buttons (added dynamically)
    document.addEventListener('click', async (e) => {
        if (e.target.closest('.save-mask-btn')) {
            const btn = e.target.closest('.save-mask-btn');
            const maskImage = btn.dataset.mask;
            const itemName = btn.dataset.name;
            const index = btn.dataset.index;
            
            if (!maskImage) return;
            
            const originalHTML = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<span style="font-size: 0.875rem;">Saving...</span>';
            
            try {
                // Compress the mask image before saving
                const compressedMask = await compressImage(maskImage, 800, 0.8);
                const fileName = `${itemName.toLowerCase().replace(/\s+/g, '_')}_mask_${Date.now()}.png`;
                
                await addToWardrobe(compressedMask, fileName);
                
                // Success feedback
                btn.innerHTML = '<span style="font-size: 0.875rem;">✓ Saved!</span>';
                btn.style.backgroundColor = '#10b981';
                btn.style.borderColor = '#10b981';
                btn.style.color = 'white';
                
                toast.success('Saved to wardrobe', `"${itemName}" mask added to your wardrobe!`);
                
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.style.backgroundColor = '';
                    btn.style.borderColor = '';
                    btn.style.color = '';
                    btn.disabled = false;
                }, 2000);
                
            } catch (error) {
                console.error('Error saving mask to wardrobe:', error);
                const hint = getErrorHint(error);
                toast.error('Save failed', `Couldn't save mask to wardrobe.${hint ? ' ' + hint : ''}`);
                btn.innerHTML = originalHTML;
                btn.disabled = false;
            }
        }
    });
}

// Helper function to save a result to history (now uses IndexedDB)
export async function saveResultToHistory(result) {
    if (!result) return false;
    
    try {
        // Check if this exact result is already saved (by timestamp)
        const existingItem = await getHistoryItemDB(result.timestamp);
        
        if (!existingItem) {
            await addToHistoryDB(result);
            return true;
        }
        
        return false; // Already saved
    } catch (e) {
        console.error('Failed to save to history:', e);
        const hint = getErrorHint(e);
        toast.error('Save failed', `Couldn't save to history.${hint ? ' ' + hint : ''}`);
        return false;
    }
}

export function displayResults(updateURL = true) {
    // Store the result for page refresh
    if (state.currentResult) {
        if (state.isFromHistory) {
            // For history items, just store the timestamp
            sessionStorage.setItem('lastViewedResultTimestamp', state.currentResult.timestamp);
            sessionStorage.removeItem('currentResult');
        } else {
            // For new analyses, store the full result
            sessionStorage.setItem('currentResult', JSON.stringify(state.currentResult));
            sessionStorage.removeItem('lastViewedResultTimestamp');
        }
    }
    
    // Toggle Save button visibility based on whether result came from history
    const saveToHistoryBtn = document.getElementById('saveToHistory');
    if (saveToHistoryBtn) {
        saveToHistoryBtn.style.display = state.isFromHistory ? 'none' : '';
    }
    
    // Update the results heading to include the occasion
    const resultsHeading = document.querySelector('#results-view .section-header h2');
    if (resultsHeading && state.currentResult && state.currentResult.context) {
        resultsHeading.textContent = `${toTitleCase(state.currentResult.context)} Style Recommendation`;
    }
    
    // Check for non-clothing items and show a note if any exist
    const nonClothingCount = state.currentResult.attributes.filter(attr => attr.isClothing === false).length;
    const existingNote = document.querySelector('#results-view .non-clothing-note');
    if (existingNote) {
        existingNote.remove();
    }
    if (nonClothingCount > 0) {
        const noteEl = document.createElement('p');
        noteEl.className = 'non-clothing-note';
        if (nonClothingCount === 1) {
            noteEl.textContent = `1 image was not detected as clothing and has been excluded from recommendations.`;
        } else {
            noteEl.textContent = `${nonClothingCount} images were not detected as clothing and have been excluded from recommendations.`;
        }
        // Insert after the section header, before the results section
        const sectionHeader = document.querySelector('#results-view .section-header');
        if (sectionHeader && sectionHeader.nextSibling) {
            sectionHeader.parentNode.insertBefore(noteEl, sectionHeader.nextSibling);
        }
    }
    
    // Update back/edit button based on source
    const backBtn = document.getElementById('backToUpload');
    if (backBtn) {
        if (state.analysisSource === 'history') {
            // Show "Back" with arrow icon
            backBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
                Back
            `;
        } else {
            // Show "Edit" with pencil icon for wardrobe/upload
            backBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                </svg>
                Edit
            `;
        }
    }
    
    // Hide loading
    hideLoading();

    // Populate results
    const llmRecommendation = document.getElementById('llmRecommendation');
    const attributesPanel = document.getElementById('attributesPanel');
    const selectedItemsDisplay = document.getElementById('selectedItemsDisplay');

    // Convert markdown to HTML to preserve subsection headers
    const htmlRecommendation = markdownToHTML(state.currentResult.recommendation);
    llmRecommendation.innerHTML = htmlRecommendation;

    // Display selected items with images - selected items first (only if attributes are available)
    if (selectedItemsDisplay) {
        if (state.currentResult.attributes && state.currentResult.attributes.length > 0) {
            console.log(`[DEBUG results.js] Displaying ${state.currentResult.attributes.length} attributes`);
            console.log('[DEBUG results.js] Images array length:', state.currentResult.images?.length || 0);
            selectedItemsDisplay.innerHTML = '<h3>Selected Items for This Occasion</h3>';
            const imagesGrid = document.createElement('div');
            imagesGrid.className = 'selected-images-grid';
        
        // Create array with index information for sorting - iterate over attributes instead of images
        const imagesWithIndex = state.currentResult.attributes.map((attr, index) => {
            const isNotClothing = attr.isClothing === false;
            // Use maskImage if available, otherwise fall back to original images array
            const imgUrl = attr.maskImage || (state.currentResult.images && state.currentResult.images[index]);
            return {
                imgUrl,
                index,
                isSelected: state.currentResult.selectedItems.includes(index),
                attr,
                isNotClothing,
                isLowConfidence: !isNotClothing && (attr.confidence <= 0.1 || attr.name === 'Unknown' || attr.name === 'Clothing Item')
            };
        });
        
        // Sort: selected items first, non-clothing items last
        imagesWithIndex.sort((a, b) => {
            // Non-clothing items always go to the end
            if (a.isNotClothing && !b.isNotClothing) return 1;
            if (!a.isNotClothing && b.isNotClothing) return -1;
            // Low confidence items go after normal items
            if (a.isLowConfidence && !b.isLowConfidence) return 1;
            if (!a.isLowConfidence && b.isLowConfidence) return -1;
            // Among normal items, selected items first
            if (a.isSelected && !b.isSelected) return -1;
            if (!a.isSelected && b.isSelected) return 1;
            return 0; // Keep original order for items in the same group
        });
        
        // Display sorted images
        imagesWithIndex.forEach(({ imgUrl, index, isSelected, attr, isNotClothing }) => {
            const imgDiv = document.createElement('div');
            imgDiv.className = `selected-image-item ${isNotClothing ? 'not-clothing' : (isSelected ? 'recommended' : 'not-recommended')}`;
            
            let badge = '';
            if (isNotClothing) {
                badge = '<div class="not-clothing-badge">Not Clothing</div>';
            } else if (isSelected) {
                badge = '<div class="selected-badge">✓ Selected</div>';
            } else {
                badge = '<div class="not-selected-badge">Not for this occasion</div>';
            }
            
            // Check if this item has a mask image (imgUrl was already computed from maskImage or images array)
            const hasMask = attr.maskImage && attr.maskImage.length > 0;
            
            imgDiv.innerHTML = `
                <div class="image-wrapper">
                    <img src="${imgUrl}" alt="Item ${index + 1}" class="${hasMask ? 'mask-image' : ''}">
                    ${badge}
                    ${hasMask ? '<div class="mask-badge">Segmented</div>' : ''}
                </div>
                <div class="image-label">
                    <strong>${formatItemName(attr.name) || 'Item ' + (index + 1)}</strong>
                    ${attr.color && attr.color !== 'N/a' ? `<span>${formatItemName(attr.color)}</span>` : ''}
                </div>
                ${!isNotClothing && hasMask ? `
                    <button class="save-mask-btn" data-index="${index}" data-mask="${imgUrl}" data-name="${attr.name || 'Item'}">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                            <polyline points="17 21 17 13 7 13 7 21"/>
                            <polyline points="7 3 7 8 15 8"/>
                        </svg>
                        Save to Wardrobe
                    </button>
                ` : ''}
            `;
            
            imagesGrid.appendChild(imgDiv);
        });
        
            selectedItemsDisplay.appendChild(imagesGrid);
        } else {
            // No images available (e.g., viewing from history)
            selectedItemsDisplay.innerHTML = '';
        }
    }

    // Display attributes - selected items first
    attributesPanel.innerHTML = '';
    
    // Create array with index information for sorting
    const attributesWithIndex = state.currentResult.attributes.map((attr, index) => {
        const isNotClothing = attr.isClothing === false;
        return {
            attr,
            index,
            isSelected: state.currentResult.selectedItems.includes(index),
            isNotClothing,
            isLowConfidence: !isNotClothing && (attr.confidence <= 0.1 || attr.name === 'Unknown' || attr.name === 'Clothing Item')
        };
    });
    
    // Sort: selected items first, non-clothing items last
    attributesWithIndex.sort((a, b) => {
        // Non-clothing items always go to the end
        if (a.isNotClothing && !b.isNotClothing) return 1;
        if (!a.isNotClothing && b.isNotClothing) return -1;
        // Low confidence items go after normal items
        if (a.isLowConfidence && !b.isLowConfidence) return 1;
        if (!a.isLowConfidence && b.isLowConfidence) return -1;
        // Among normal items, selected items first
        if (a.isSelected && !b.isSelected) return -1;
        if (!a.isSelected && b.isSelected) return 1;
        return 0; // Keep original order for items in the same group
    });
    
    // Display sorted attributes
    attributesWithIndex.forEach(({ attr, index, isSelected, isNotClothing }) => {
        const div = document.createElement('div');
        let attrClass = 'attribute-item';
        if (isNotClothing) {
            attrClass += ' not-clothing-attribute';
        } else if (isSelected) {
            attrClass += ' selected-attribute';
        } else {
            attrClass += ' valid-attribute';
        }
        div.className = attrClass;
        
        const confidence = Math.round(attr.confidence * 100);
        
        let badge = '';
        if (isNotClothing) {
            badge = '<span class="badge-mini-warning">Not Clothing</span>';
        } else if (isSelected) {
            badge = '<span class="badge-mini">✓ Selected</span>';
        } else {
            badge = '<span class="badge-mini-gray">Not selected</span>';
        }
        
        div.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4>${attr.category || 'Item'}</h4>
                ${badge}
            </div>
            <div class="attribute-value">${formatItemName(attr.name) || 'Unknown'}</div>
            <div class="confidence">
                <span>Detection: ${confidence}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                </div>
            </div>
            ${attr.texture && attr.texture !== 'N/a' ? `<p style="margin-top: 0.5rem; color: var(--primary-light); font-size: 0.875rem;">Texture: ${formatItemName(attr.texture)}</p>` : ''}
            ${attr.color && attr.color !== 'N/a' ? `<p style="color: var(--primary-light); font-size: 0.875rem;">Color: ${formatItemName(attr.color)}</p>` : ''}
        `;
        
        attributesPanel.appendChild(div);
    });

    // Switch to results view and update URL (if requested)
    switchView('results', updateURL);
    // Show upload section if it exists (for regular upload flow)
    const uploadSection = document.querySelector('.upload-section');
    if (uploadSection) {
        uploadSection.style.display = 'block';
    }
}
