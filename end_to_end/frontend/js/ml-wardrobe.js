// ML Backend Wardrobe Analysis

import { state } from './state.js';
import { toast } from './toast.js';
import { toTitleCase, getErrorHint } from './utils.js';
import { getAllWardrobeItems } from './db.js';

/**
 * Get recommendation from wardrobe using ML backend
 * This bypasses the regular upload flow and sends wardrobe data directly to ML backend
 */
export async function getWardrobeRecommendationML(onDisplayResults) {
    try {
        // Get selected wardrobe items
        const allItems = await getAllWardrobeItems();
        const selectedItems = state.selectedWardrobeItems
            .map(id => allItems.find(item => item.id === id))
            .filter(item => item);
        
        if (selectedItems.length === 0) {
            throw new Error('No wardrobe items selected');
        }
        
        const context = toTitleCase(state.wardrobeCustomContext || state.wardrobeContext);
        
        console.log(`Analyzing ${selectedItems.length} wardrobe items with ML backend for: ${context}`);
        
        // Prepare wardrobe items data
        const wardrobeItems = selectedItems.map(item => ({
            id: item.id,
            imageData: item.imageData, // base64 data URL
            fileName: item.fileName
        }));
        
        // Call ML backend wardrobe analysis endpoint
        const response = await fetch('http://localhost:3000/api/ml/analyze-wardrobe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                wardrobeItems,
                context
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(`ML wardrobe analysis failed: ${errorData.message || errorData.error}`);
        }
        
        const mlData = await response.json();
        
        // Convert ML predictions to frontend-compatible attributes format
        const attributes = mlData.predictions.map((pred, idx) => ({
            isClothing: true,
            name: `${pred.gender} ${pred.baseColour} ${pred.articleType}`.trim(),
            color: pred.baseColour,
            texture: 'Unknown',
            category: pred.usage,
            confidence: 0.85,
            mlPredictions: pred
        }));
        
        // Store result
        state.currentResult = {
            context: context,
            attributes: attributes,
            recommendation: mlData.recommendation,
            selectedItems: mlData.selectedItems || [],
            images: selectedItems.map(item => item.imageData),
            timestamp: new Date().toISOString(),
            visionOutput: mlData.visionOutput // Store for debugging
        };
        
        state.isFromHistory = false;
        state.analysisSource = 'wardrobe';
        
        // Display results
        if (onDisplayResults) {
            onDisplayResults();
        }
        
        return mlData;
        
    } catch (error) {
        console.error('Error in ML wardrobe analysis:', error);
        
        const hint = getErrorHint(error);
        toast.error(
            'Wardrobe analysis failed',
            hint || 'Make sure the ML backend server is running.',
            6000
        );
        
        throw error;
    }
}
