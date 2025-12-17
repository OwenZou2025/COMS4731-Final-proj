// Main Application Entry Point

import { state } from './state.js';
import { toast } from './toast.js';
import { initDB, migrateHistoryToIndexedDB } from './db.js';
import { 
    initNavigation, 
    initializeView, 
    getViewFromURL, 
    switchView,
    registerViewCallback 
} from './navigation.js';
import { 
    initUpload, 
    updateSubmitButton, 
    updateItemCounter, 
    resetUploadForm,
    setGetRecommendationCallback 
} from './upload.js';
import { 
    initAnalysis, 
    getRecommendation,
    setDisplayResultsCallback 
} from './analysis.js';
import { initResults, displayResults } from './results.js';
import { initHistory, loadHistory, getHistory } from './history.js';
import { 
    initWardrobe, 
    loadWardrobe, 
    setupWardrobeContextListeners,
    updateSelectedWardrobeDisplay,
    updateWardrobeSubmitButton 
} from './wardrobe.js';

// Backend toggle functionality
function initBackendToggle() {
    const toggleBtn = document.getElementById('backendToggle');
    const backendName = document.getElementById('backendName');
    
    if (!toggleBtn || !backendName) return;
    
    toggleBtn.addEventListener('click', () => {
        state.useMLBackend = !state.useMLBackend;
        updateBackendUI();
        
        const backendType = state.useMLBackend ? 'Custom ML Backend' : 'Gemini AI';
        toast.success('Backend switched', `Now using: ${backendType}`, 2000);
    });
    
    // Initial UI state
    updateBackendUI();
}

function updateBackendUI() {
    const toggleBtn = document.getElementById('backendToggle');
    const backendName = document.getElementById('backendName');
    
    if (!toggleBtn || !backendName) return;
    
    if (state.useMLBackend) {
        backendName.textContent = 'Custom ML';
        toggleBtn.classList.add('ml-active');
    } else {
        backendName.textContent = 'Gemini';
        toggleBtn.classList.remove('ml-active');
    }
}

// Initialize database first, then view (history now requires DB)
(async function initApp() {
    // Initialize toast system first
    toast.init();
    
    // Wire up callbacks to avoid circular dependencies
    setGetRecommendationCallback(getRecommendation);
    setDisplayResultsCallback(displayResults);
    
    // Register view callbacks for navigation
    registerViewCallback('upload', {
        reset: resetUploadForm
    });
    
    registerViewCallback('wardrobe', {
        load: loadWardrobe,
        setupContextListeners: setupWardrobeContextListeners,
        updateSelectedDisplay: updateSelectedWardrobeDisplay,
        updateSubmitButton: updateWardrobeSubmitButton
    });
    
    registerViewCallback('history', {
        load: loadHistory,
        getAll: getHistory
    });
    
    registerViewCallback('results', {
        display: displayResults
    });
    
    try {
        await initDB();
        console.log('[init] DB ready');
        
        // Migrate any existing localStorage history to IndexedDB
        await migrateHistoryToIndexedDB();
        
        // Initialize all modules
        initNavigation();
        initUpload();
        initAnalysis();
        initResults();
        initHistory();
        initWardrobe();
        
        // Now initialize view (can access history from IndexedDB)
        await initializeView();
        
        updateSubmitButton();
        updateItemCounter();
        
        const clearSelectionBtn = document.getElementById('clearWardrobe');
        if (clearSelectionBtn) {
            clearSelectionBtn.style.display = 'none';
        }
        
        // Reload wardrobe if we're on the wardrobe view
        if (getViewFromURL() === 'wardrobe') {
            loadWardrobe();
            setupWardrobeContextListeners();
        }
        
        // Reload history if we're on the history view
        if (getViewFromURL() === 'history') {
            await loadHistory();
        }
        
        // Initialize backend toggle
        initBackendToggle();
        
    } catch (error) {
        console.error('Failed to initialize database:', error);
        // Show a non-intrusive warning instead of blocking alert
        console.warn('Some features may not work due to storage issues. Try closing other tabs or clearing browser data.');
        
        // Initialize modules for basic functionality
        initNavigation();
        initUpload();
        initAnalysis();
        initResults();
        initHistory();
        initWardrobe();
        
        // Still try to initialize view for basic functionality
        switchView(getViewFromURL(), false);
        updateSubmitButton();
        updateItemCounter();
    }
})();
