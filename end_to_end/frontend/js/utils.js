// Utility functions and helpers

// Smart error message helper - adds contextual hints based on error type
export function getErrorHint(error) {
    if (!error) return '';
    
    const errorStr = error.toString().toLowerCase();
    const errorName = error.name?.toLowerCase() || '';
    
    // Network/fetch errors
    if (error instanceof TypeError && errorStr.includes('fetch')) {
        return 'Check your internet connection.';
    }
    if (errorStr.includes('networkerror') || errorStr.includes('network')) {
        return 'Check your internet connection.';
    }
    
    // Server errors (5xx)
    if (error.status >= 500 || errorStr.includes('500') || errorStr.includes('502') || 
        errorStr.includes('503') || errorStr.includes('server error')) {
        return 'Server error. Try again in a moment.';
    }
    
    // IndexedDB/Storage errors
    if (errorName.includes('quota') || errorStr.includes('quota')) {
        return 'Storage full. Try clearing some history.';
    }
    if (errorStr.includes('blocked') || errorStr.includes('security') || 
        errorStr.includes('access') && errorStr.includes('denied')) {
        return 'Storage may be blocked. Try refreshing the page.';
    }
    if (errorStr.includes('transaction') || errorStr.includes('database') && errorStr.includes('closed')) {
        return 'Another session may be open. Close extra tabs and try again.';
    }
    
    // No specific hint detected
    return '';
}

// Common abbreviated occasions/places for fashion contexts
const ABBREVIATIONS = new Set([
    'nyc', 'la', 'uk', 'usa',
    'nye', 'bbq', 'vip', 'nyfw',
    'pfw', 'cfda', 'gq',
]);

// Helper function to title-case strings (e.g., "birthday party" -> "Birthday Party")
export function toTitleCase(str) {
    if (!str) return str;
    
    return str.replace(/\w\S*/g, (txt) => {
        const lower = txt.toLowerCase();
        if (ABBREVIATIONS.has(lower)) {
            return txt.toUpperCase();
        }
        return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    });
}

// Function to format item names consistently (title case)
export function formatItemName(name) {
    if (!name) return '';
    // Convert to title case: capitalize first letter of each word
    return name
        .toLowerCase()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Function to convert markdown to HTML (for style assessment subsections)
export function markdownToHTML(text) {
    if (!text) return '';
    
    // Split into lines for processing
    const lines = text.split('\n');
    let html = '';
    let inList = false;
    let listType = null; // 'ul' or 'ol'
    
    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        
        // Check for subsection headers
        const headerMatch = line.match(/^\*\*(Overall Assessment|Why These Pieces Work|Outfit Combinations|Additional Styling Tips)\*\*$/i);
        if (headerMatch) {
            if (inList) {
                html += `</${listType}>`;
                inList = false;
            }
            html += `<h4 class="subsection-header">${toTitleCase(headerMatch[1])}</h4>`;
            continue;
        }
        
        // Check for bullet points (-, •, or *)
        const bulletMatch = line.match(/^[-•\*]\s+(.+)$/);
        if (bulletMatch) {
            if (!inList || listType !== 'ul') {
                if (inList) html += `</${listType}>`;
                html += '<ul>';
                inList = true;
                listType = 'ul';
            }
            let content = bulletMatch[1]
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
            html += `<li>${content}</li>`;
            continue;
        }
        
        // Check for numbered lists (1. 2. etc)
        const numberedMatch = line.match(/^\d+\.\s+(.+)$/);
        if (numberedMatch) {
            if (!inList || listType !== 'ol') {
                if (inList) html += `</${listType}>`;
                html += '<ol>';
                inList = true;
                listType = 'ol';
            }
            let content = numberedMatch[1]
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
            html += `<li>${content}</li>`;
            continue;
        }
        
        // Regular text - close any open list first
        if (inList) {
            html += `</${listType}>`;
            inList = false;
        }
        
        // Skip empty lines but add paragraph break
        if (line === '') {
            continue;
        }
        
        // Process inline markdown for regular paragraphs
        let processed = line
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/__(.*?)__/g, '<strong>$1</strong>')
            .replace(/_(.*?)_/g, '<em>$1</em>');
        
        html += `<p>${processed}</p>`;
    }
    
    // Close any remaining open list
    if (inList) {
        html += `</${listType}>`;
    }
    
    return html;
}

// Compress image data URL to reduce storage size
export async function compressImage(dataURL, maxWidth = 800, quality = 0.8) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        
        img.onload = () => {
            // Calculate new dimensions maintaining aspect ratio
            let width = img.width;
            let height = img.height;
            
            if (width > maxWidth) {
                height = Math.round((height * maxWidth) / width);
                width = maxWidth;
            }
            
            // Create canvas and draw resized image
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);
            
            // Convert to compressed JPEG
            const compressedDataURL = canvas.toDataURL('image/jpeg', quality);
            resolve(compressedDataURL);
        };
        
        img.onerror = () => {
            reject(new Error('Failed to load image for compression'));
        };
        
        img.src = dataURL;
    });
}

// Get storage usage estimate
export async function getStorageEstimate() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
        try {
            const estimate = await navigator.storage.estimate();
            const usageInMB = (estimate.usage / (1024 * 1024)).toFixed(2);
            const quotaInMB = (estimate.quota / (1024 * 1024)).toFixed(2);
            const percentUsed = ((estimate.usage / estimate.quota) * 100).toFixed(1);
            
            return {
                usage: usageInMB,
                quota: quotaInMB,
                percentUsed: percentUsed,
                available: (estimate.quota - estimate.usage) / (1024 * 1024)
            };
        } catch (err) {
            console.warn('Could not estimate storage:', err);
            return null;
        }
    }
    return null;
}
