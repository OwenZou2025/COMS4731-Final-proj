// Gemini AI Integration

const { GoogleGenerativeAI } = require('@google/generative-ai');
const { generateMockAttributes, generateMockRecommendation } = require('./mock');

// Check demo mode status
function checkDemoMode() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === '') return true;
    
    // Check for common placeholder values
    const placeholders = [
        'YOUR_API_KEY_HERE',
        'your_api_key_here',
        'your_actual_api_key_here',
        'INSERT_YOUR_KEY_HERE',
        'REPLACE_WITH_YOUR_KEY',
        'YOUR-API-KEY',
        'YOUR_GEMINI_API_KEY'
    ];
    
    const keyLower = apiKey.toLowerCase();
    return placeholders.some(p => keyLower === p.toLowerCase());
}

const DEMO_MODE = checkDemoMode();

// Initialize Gemini API (only if not in demo mode)
const genAI = DEMO_MODE ? null : new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

/**
 * Check if running in demo mode
 * @returns {boolean}
 */
function isDemoMode() {
    return DEMO_MODE;
}

/**
 * Analyze images using Gemini Vision (one call per image, in parallel)
 * @param {Buffer[]} imageBuffers - Array of image buffers
 * @returns {Promise<Array>} Array of clothing attributes
 */
async function analyzeImagesWithGemini(imageBuffers) {
    if (DEMO_MODE) {
        console.log('Running in DEMO MODE - Using mock data');
        return generateMockAttributes(imageBuffers.length);
    }

    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

    // Helper to analyze a single image with retry logic
    async function analyzeSingleImage(buffer, index, retries = 2) {
        const imagePart = {
            inlineData: {
                data: buffer.toString("base64"),
                mimeType: "image/jpeg"
            }
        };

        const prompt = `You are a fashion classifier.

You are given a **single** image (Image ${index + 1}). 
First, determine if this image contains a clothing item, footwear, or fashion accessory (like bags, hats, scarves, jewelry).

Return a **single JSON object** with this exact structure:

{
  "isClothing": true,
  "name": "short, specific garment name",
  "color": "primary visible color",
  "texture": "main material or texture",
  "category": "style category (casual | formal | business | athletic | streetwear | evening | loungewear)",
  "confidence": 0.0
}

Rules:
- Set "isClothing" to true ONLY if the image shows clothing, footwear, or fashion accessories.
- Set "isClothing" to false if the image shows: vehicles, animals, food, buildings, landscapes, electronics, furniture, people without clear focus on their clothing, or any other non-fashion item.
- If "isClothing" is false, set name to "None", color to "N/a", texture to "N/a", category to "N/A", and confidence to 0.
- Only describe THIS image, not any others.
- Pick one dominant garment (e.g., if it shows a blazer and pants, describe the blazer).
- "confidence" must be a number between 0 and 1.
- Return **JSON only**, no extra text.`;

        try {
            const result = await model.generateContent({
                contents: [
                    {
                        role: "user",
                        parts: [
                            { text: prompt },
                            imagePart
                        ]
                    }
                ]
            });

            const text = result.response.text();
            console.log(`Gemini response for image ${index + 1}:`, text);

            let attrs;
            try {
                // Try direct JSON parse first
                attrs = JSON.parse(text);
            } catch {
                // Fallback: extract first JSON object from the text
                const match = text.match(/\{[\s\S]*\}/);
                if (match) {
                    attrs = JSON.parse(match[0]);
                } else {
                    console.log(`Could not parse JSON for image ${index + 1}, using fallback.`);
                    attrs = {
                        isClothing: false,
                        name: "None",
                        color: "N/a",
                        texture: "N/a",
                        category: "N/A",
                        confidence: 0
                    };
                }
            }
            
            // Ensure isClothing field exists (backwards compatibility)
            if (attrs.isClothing === undefined) {
                attrs.isClothing = attrs.confidence > 0.1 && attrs.name !== "None" && attrs.name !== "Unknown";
            }

            // Attach index so frontend can always map correctly
            return {
                imageIndex: index,
                ...attrs
            };
        } catch (error) {
            // Retry on transient network errors
            const isTransientError = error.message?.includes('fetch failed') || 
                                     error.message?.includes('ECONNRESET') ||
                                     error.message?.includes('ETIMEDOUT') ||
                                     error.code === 'ECONNRESET';
            
            if (isTransientError && retries > 0) {
                console.log(`Retrying image ${index + 1} after transient error (${retries} retries left)...`);
                // Wait a bit before retrying (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, 1000 * (3 - retries)));
                return analyzeSingleImage(buffer, index, retries - 1);
            }

            console.error(`Error analyzing image ${index + 1}:`, error);

            // Return a safe fallback for this image so Promise.all still resolves
            return {
                imageIndex: index,
                isClothing: false,
                name: "None",
                color: "N/a",
                texture: "N/a",
                category: "N/A",
                confidence: 0
            };
        }
    }

    // Run all image analyses in parallel
    const promises = imageBuffers.map((buffer, index) =>
        analyzeSingleImage(buffer, index)
    );

    const allAttributes = await Promise.all(promises);

    // Sort by imageIndex to keep the array ordered
    allAttributes.sort((a, b) => a.imageIndex - b.imageIndex);

    return allAttributes;
}

/**
 * Get style recommendations from Gemini
 * @param {Array} attributes - Clothing attributes
 * @param {string} context - Occasion/context
 * @returns {Promise<Object>} Recommendation with selectedItems and text
 */
async function getRecommendationFromGemini(attributes, context) {
    // Demo mode - return mock recommendation
    if (DEMO_MODE) {
        console.log('Running in DEMO MODE - using mock recommendation');
        return generateMockRecommendation(attributes, context);
    }

    try {
        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

        const attributesText = attributes.map((attr, idx) => 
            `[${idx}] ${attr.name} - Color: ${attr.color}, Texture: ${attr.texture}, Category: ${attr.category}`
        ).join('\n');

        const prompt = `You are a professional fashion stylist. I have the following clothing items:
${attributesText}

The occasion/context is: ${context}

Please analyze and provide a response in TWO parts:

PART 1 - Selected Items (JSON format):
Return a JSON array of item indices (the numbers in brackets like [0], [1], etc.) that are appropriate for this occasion.
Example: [0, 2, 3]

PART 2 - Style Recommendation (text):
Provide a detailed style assessment organized into these EXACT subsections.

CRITICAL NAMING RULE: NEVER use "Item 0", "Item 1", "Item 2" etc. NEVER write formats like "Item 0 (description):" or "Item 1 (name):". Instead, ALWAYS refer to items ONLY by their descriptive names. ALWAYS use Title Case for item names (capitalize each word), like "Navy Straight-leg Trousers", "Double-breasted Plaid Blazer", or "Cropped Tweed Jacket".

**Overall Assessment**
A brief style assessment of the SELECTED items (2-3 sentences about the overall look and aesthetic)

**Why These Pieces Work**
For each selected item, explain why it works for this occasion. Format as bullet points with the item name in BOLD (in Title Case) followed by a colon, like:
- **Navy Straight-leg Trousers:** These are a foundational piece...
- **Double-breasted Plaid Blazer:** This adds sophistication...

**Outfit Combinations**
Provide 2-3 specific outfit combination suggestions. Give each outfit a CREATIVE NAME in bold, then describe what items to combine. Format like:
1. **The Polished Professional:** Pair the Navy Straight-leg Trousers with...
2. **Chic Business Casual:** Combine the Double-breasted Blazer with...
3. **Effortless Elegance:** Layer the Cropped Tweed Jacket over...

**Additional Styling Tips**
Organize tips by CATEGORY with bold headers. Format like:
- **Tops:** Suggestions for tops to pair with these items...
- **Footwear:** Shoe recommendations...
- **Accessories:** Belt, jewelry, bag suggestions...
- **Layering:** Tips for layering pieces...

Format your response exactly as:
SELECTED_ITEMS: [array of indices]
RECOMMENDATION: your text here

Keep the recommendation natural, friendly, and conversational. Make sure to include all four subsection headers in bold (wrapped in **).`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();

        // Parse the response
        return parseRecommendationResponse(text, attributes.length);

    } catch (error) {
        console.error('Error getting recommendation from Gemini:');
        console.error('Error name:', error.name);
        console.error('Error message:', error.message);
        if (error.response) {
            console.error('Error response:', error.response);
        }
        if (error.stack) {
            console.error('Stack trace:', error.stack);
        }
        throw error;
    }
}

/**
 * Parse the recommendation response to extract selected items and text
 * @param {string} text - Raw response text
 * @param {number} totalItems - Total number of items
 * @returns {Object} Parsed recommendation
 */
function parseRecommendationResponse(text, totalItems) {
    let selectedItems = [];
    let recommendation = text;

    // Try to extract selected items
    const selectedMatch = text.match(/SELECTED_ITEMS:\s*(\[[\d,\s]*\])/);
    if (selectedMatch) {
        try {
            selectedItems = JSON.parse(selectedMatch[1]);
            // Validate indices
            selectedItems = selectedItems.filter(idx => idx >= 0 && idx < totalItems);
        } catch (e) {
            console.log('Could not parse selected items, selecting all');
        }
    }

    // Extract recommendation text
    const recMatch = text.match(/RECOMMENDATION:\s*([\s\S]*)/);
    if (recMatch) {
        recommendation = recMatch[1].trim();
    }

    // If no items selected, select all
    if (selectedItems.length === 0) {
        selectedItems = Array.from({ length: totalItems }, (_, i) => i);
    }

    return {
        selectedItems,
        recommendation
    };
}

module.exports = {
    isDemoMode,
    analyzeImagesWithGemini,
    getRecommendationFromGemini
};

