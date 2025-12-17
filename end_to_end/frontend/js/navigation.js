// Navigation and View Switching

import { toast } from './toast.js';
import { state } from './state.js';
import { getDb } from './db.js';

// DOM Elements
let navButtons;
let views;
let logoBtn;

// Callback registries for view-specific actions
const viewCallbacks = {
    upload: null,
    wardrobe: null,
    history: null,
    results: null
};

export function registerViewCallback(viewName, callback) {
    viewCallbacks[viewName] = callback;
}

export function initNavigation() {
    navButtons = document.querySelectorAll('.nav-btn');
    views = document.querySelectorAll('.view');
    logoBtn = document.getElementById('logoBtn');

    // Navigation button click handlers
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetView = btn.dataset.view;
            switchView(targetView, true);
        });
    });

    // Logo click - go to main page
    logoBtn.addEventListener('click', () => {
        switchView('upload', true);
    });

    // Handle browser back/forward buttons
    window.addEventListener('popstate', (event) => {
        const viewName = event.state?.view || getViewFromURL();
        switchView(viewName, false);
    });
}

export function switchView(viewName, updateURL = false) {
    // Dismiss any toasts when switching views
    toast.dismissAll();
    
    // Get current active view
    const currentView = Array.from(views).find(view => view.classList.contains('active'));
    const currentViewName = currentView ? currentView.id.replace('-view', '') : null;
    const isSwitchingView = currentViewName !== viewName;
    
    // Update nav buttons
    navButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    // Update views
    views.forEach(view => {
        view.classList.toggle('active', view.id === `${viewName}-view`);
    });

    // Update URL if requested (but not on initial load)
    if (updateURL) {
        const newURL = viewName === 'upload' ? '/' : `/${viewName}`;
        window.history.pushState({ view: viewName }, '', newURL);
    }

    // Reset state when switching to a different view
    if (isSwitchingView) {
        if (viewName === 'upload') {
            // Reset upload form when switching to upload view
            if (viewCallbacks.upload?.reset) {
                viewCallbacks.upload.reset();
            }
        } else if (viewName === 'wardrobe') {
            // Only clear selections if NOT coming back from a wardrobe analysis
            if (state.analysisSource !== 'wardrobe') {
                state.selectedWardrobeItems = [];
                state.wardrobeContext = null;
                state.wardrobeCustomContext = '';
                
                // Clear wardrobe context input and buttons
                const wardrobeCustomContextInput = document.getElementById('wardrobeCustomContext');
                if (wardrobeCustomContextInput) {
                    wardrobeCustomContextInput.value = '';
                }
                const wardrobeContextButtons = document.querySelectorAll('.wardrobe-context-btn');
                wardrobeContextButtons.forEach(btn => btn.classList.remove('selected'));
            }
        }
    }
    
    // Load view-specific data (always, regardless of whether we're switching)
    const db = getDb();
    if (viewName === 'wardrobe') {
        // Update wardrobe display
        if (db) {
            if (viewCallbacks.wardrobe?.load) {
                viewCallbacks.wardrobe.load();
            }
            if (viewCallbacks.wardrobe?.setupContextListeners) {
                viewCallbacks.wardrobe.setupContextListeners();
            }
            if (isSwitchingView) {
                if (viewCallbacks.wardrobe?.updateSelectedDisplay) {
                    viewCallbacks.wardrobe.updateSelectedDisplay();
                }
                if (viewCallbacks.wardrobe?.updateSubmitButton) {
                    viewCallbacks.wardrobe.updateSubmitButton();
                }
            }
        } else {
            // Show loading state while waiting for DB
            const wardrobeGrid = document.getElementById('wardrobeGrid');
            if (wardrobeGrid) {
                wardrobeGrid.innerHTML = `
                    <div class="empty-state" style="grid-column: 1 / -1;">
                        <div class="spinner" style="margin: 0 auto;"></div>
                        <p>Loading wardrobe...</p>
                    </div>
                `;
            }
        }
    } else if (viewName === 'history') {
        // Load history if switching to history view
        // Only load history if database is ready
        if (db) {
            if (viewCallbacks.history?.load) {
                viewCallbacks.history.load();
            }
        } else {
            // Show loading state while waiting for DB
            const historyList = document.getElementById('historyList');
            if (historyList) {
                historyList.innerHTML = `
                    <div class="empty-state">
                        <div class="spinner" style="margin: 0 auto;"></div>
                        <p>Loading history...</p>
                    </div>
                `;
            }
        }
    }
}

// Get view name from current URL
export function getViewFromURL() {
    const path = window.location.pathname;
    if (path === '/' || path === '') {
        return 'upload';
    }
    // Extract view name from path like /results, /history, /wardrobe
    const viewName = path.substring(1).split('/')[0];
    return ['upload', 'results', 'history', 'wardrobe'].includes(viewName) ? viewName : 'upload';
}

// Initialize view based on URL on page load
export async function initializeView() {
    const viewName = getViewFromURL();
    
    // If we're on results page but don't have currentResult, try to restore
    if (viewName === 'results' && !state.currentResult) {
        // First check for a current unsaved result
        const storedCurrentResult = sessionStorage.getItem('currentResult');
        if (storedCurrentResult) {
            try {
                state.currentResult = JSON.parse(storedCurrentResult);
                state.isFromHistory = false;
                if (viewCallbacks.results?.display) {
                    viewCallbacks.results.display(false);
                }
                return;
            } catch (e) {
                console.error('Failed to parse stored result:', e);
                sessionStorage.removeItem('currentResult');
            }
        }
        
        // Then try to restore from history (now async)
        if (viewCallbacks.history?.getAll) {
            const history = await viewCallbacks.history.getAll();
            if (history.length > 0) {
                // Try to restore the specific result by timestamp from sessionStorage
                const lastViewedTimestamp = sessionStorage.getItem('lastViewedResultTimestamp');
                if (lastViewedTimestamp) {
                    const specificResult = history.find(item => item.timestamp == lastViewedTimestamp);
                    if (specificResult) {
                        state.currentResult = specificResult;
                        state.isFromHistory = true; // Mark as from history
                        state.analysisSource = 'history'; // Set source for back button
                        if (viewCallbacks.results?.display) {
                            viewCallbacks.results.display(false);
                        }
                        return;
                    }
                }
                // Fallback: Restore the most recent result
                state.currentResult = history[0];
                state.isFromHistory = true; // Mark as from history
                state.analysisSource = 'history'; // Set source for back button
                // Display results without updating URL (we're already on /results)
                if (viewCallbacks.results?.display) {
                    viewCallbacks.results.display(false);
                }
                return;
            } else {
                // No results available, redirect to upload
                switchView('upload', true);
                return;
            }
        }
    }
    
    switchView(viewName, false);
}
