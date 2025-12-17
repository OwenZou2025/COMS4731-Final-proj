// History Management and UI

import { state } from './state.js';
import { toast } from './toast.js';
import { toTitleCase, getErrorHint } from './utils.js';
import { getAllHistoryDB, deleteFromHistoryDB, clearAllHistoryDB } from './db.js';
import { displayResults } from './results.js';

// History Confirmation Modal
let historyActionCallback = null;

export function initHistory() {
    // History modal event listeners
    const cancelBtn = document.getElementById('cancelHistoryAction');
    const confirmBtn = document.getElementById('confirmHistoryAction');
    
    if (cancelBtn) {
        cancelBtn.addEventListener('click', hideHistoryConfirmModal);
    }
    
    if (confirmBtn) {
        confirmBtn.addEventListener('click', async () => {
            if (historyActionCallback) {
                await historyActionCallback();
            }
            hideHistoryConfirmModal();
        });
    }

    // Clear history button
    const clearHistoryBtn = document.getElementById('clearHistory');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', async () => {
            showHistoryConfirmModal(
                'Clear all history?',
                'This action cannot be undone.',
                'Clear All',
                async () => {
                    try {
                        await clearAllHistoryDB();
                        sessionStorage.removeItem('lastViewedResultTimestamp');
                        sessionStorage.removeItem('currentResult');
                        state.currentResult = null;
                        await loadHistory();
                    } catch (e) {
                        console.error('Failed to clear history:', e);
                        const hint = getErrorHint(e);
                        toast.error('Clear failed', `Couldn't clear history.${hint ? ' ' + hint : ''}`);
                    }
                }
            );
        });
    }
}

export async function getHistory() {
    try {
        return await getAllHistoryDB();
    } catch (e) {
        console.error('Failed to get history:', e);
        return [];
    }
}

export async function loadHistory() {
    const historyList = document.getElementById('historyList');
    
    // Show loading state
    historyList.innerHTML = `<div class="empty-state"><p>Loading history...</p></div>`;
    
    const history = await getHistory();

    if (history.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v20M2 12h20"/>
                </svg>
                <h3>No history yet</h3>
                <p>Your previous analyses will appear here</p>
            </div>
        `;
        return;
    }

    historyList.innerHTML = '';
    history.forEach((item, index) => {
        const date = new Date(item.timestamp);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Get preview text - strip all markdown headers and get the actual content
        let previewText = item.recommendation || '';
        
        // Remove all section headers
        previewText = previewText
            .replace(/\*\*Overall assessment\*\*/gi, '')
            .replace(/\*\*Why these pieces work\*\*/gi, '')
            .replace(/\*\*Outfit combinations\*\*/gi, '')
            .replace(/\*\*Additional styling tips\*\*/gi, '')
            .replace(/Overall assessment/gi, '')
            .trim();
        
        // Get first 150 characters of actual content
        if (previewText.length > 150) {
            previewText = previewText.substring(0, 150) + '...';
        }
        
        // Get thumbnail from first selected item (if available), otherwise fall back to first image
        let thumbnailUrl = null;
        if (item.images && item.images.length > 0) {
            if (item.selectedItems && item.selectedItems.length > 0) {
                // Use the first selected item's image as thumbnail
                const firstSelectedIndex = item.selectedItems[0];
                thumbnailUrl = item.images[firstSelectedIndex] || item.images[0];
            } else {
                thumbnailUrl = item.images[0];
            }
        }
        
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            ${thumbnailUrl ? `<div class="history-thumbnail"><img src="${thumbnailUrl}" alt="Preview"></div>` : ''}
            <div class="history-content">
                <div class="history-header">
                    <span class="history-date">${dateStr}</span>
                    <button class="btn-delete-history" data-timestamp="${item.timestamp}" title="Delete this entry">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
                <div class="history-preview"><strong>${toTitleCase(item.context)}</strong><br>${previewText}</div>
            </div>
        `;
        
        // Add click handler for the history item (not the delete button)
        div.addEventListener('click', (e) => {
            // Don't trigger if clicking the delete button
            if (e.target.closest('.btn-delete-history')) return;
            
            state.currentResult = item;
            state.isFromHistory = true; // Mark as from history
            state.analysisSource = 'history';
            displayResults();
            // URL will be updated by displayResults -> switchView('results', true)
        });
        
        // Add click handler for the delete button
        const deleteBtn = div.querySelector('.btn-delete-history');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent triggering the item click
            deleteHistoryItem(item.timestamp);
        });
        
        historyList.appendChild(div);
    });
}

function showHistoryConfirmModal(title, description, confirmText, callback) {
    const modal = document.getElementById('historyConfirmModal');
    const titleEl = document.getElementById('historyModalTitle');
    const descEl = document.getElementById('historyModalDescription');
    const confirmBtn = document.getElementById('confirmHistoryAction');
    
    titleEl.textContent = title;
    descEl.textContent = description;
    confirmBtn.textContent = confirmText;
    historyActionCallback = callback;
    
    modal.classList.remove('hidden');
}

function hideHistoryConfirmModal() {
    const modal = document.getElementById('historyConfirmModal');
    modal.classList.add('hidden');
    historyActionCallback = null;
}

async function deleteHistoryItem(timestamp) {
    showHistoryConfirmModal(
        'Delete this analysis?',
        'This action cannot be undone.',
        'Delete',
        async () => {
            try {
                await deleteFromHistoryDB(timestamp);
                await loadHistory();
                
                // If the deleted item was being viewed, clear it
                if (state.currentResult && state.currentResult.timestamp === timestamp) {
                    state.currentResult = null;
                    sessionStorage.removeItem('lastViewedResultTimestamp');
                    sessionStorage.removeItem('currentResult');
                }
            } catch (e) {
                console.error('Failed to delete history item:', e);
                const hint = getErrorHint(e);
                toast.error('Delete failed', `Couldn't remove from history.${hint ? ' ' + hint : ''}`);
            }
        }
    );
}
