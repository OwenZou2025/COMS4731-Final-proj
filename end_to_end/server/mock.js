// Demo Mode Mock Data Generators

/**
 * Generate mock clothing attributes for demo mode
 * @param {number} count - Number of mock items to generate
 * @returns {Array} Array of mock clothing attributes
 */
function generateMockAttributes(count) {
    const mockItems = [
        { isClothing: true, name: "Blue Denim Shirt", color: "Blue", texture: "Denim", category: "Casual", confidence: 0.92 },
        { isClothing: true, name: "Black Blazer", color: "Black", texture: "Wool", category: "Formal", confidence: 0.95 },
        { isClothing: true, name: "White Sneakers", color: "White", texture: "Leather", category: "Athletic", confidence: 0.88 },
        { isClothing: true, name: "Khaki Chinos", color: "Beige", texture: "Cotton", category: "Business Casual", confidence: 0.91 },
        { isClothing: true, name: "Navy Polo Shirt", color: "Navy", texture: "Cotton", category: "Casual", confidence: 0.89 }
    ];
    
    return mockItems.slice(0, Math.min(count, mockItems.length)).map((item, index) => ({
        imageIndex: index,
        ...item
    }));
}

/**
 * Generate mock style recommendation for demo mode
 * @param {Array} attributes - Clothing attributes
 * @param {string} context - Occasion/context
 * @returns {Object} Mock recommendation with selectedItems and recommendation text
 */
function generateMockRecommendation(attributes, context) {
    const itemsList = attributes.map(attr => attr.name).join(', ');
    
    // Intelligently select items based on context
    let selectedItems = [];
    if (context.toLowerCase().includes('formal') || context.toLowerCase().includes('office')) {
        // Select formal/business items
        selectedItems = attributes.map((attr, idx) => 
            attr.category.toLowerCase().includes('formal') || 
            attr.category.toLowerCase().includes('business') ? idx : -1
        ).filter(idx => idx !== -1);
    } else if (context.toLowerCase().includes('athletic')) {
        // Select athletic items
        selectedItems = attributes.map((attr, idx) => 
            attr.category.toLowerCase().includes('athletic') ? idx : -1
        ).filter(idx => idx !== -1);
    } else {
        // For casual or other contexts, select casual items
        selectedItems = attributes.map((attr, idx) => 
            attr.category.toLowerCase().includes('casual') ? idx : -1
        ).filter(idx => idx !== -1);
    }

    // If no matches, select first 2-3 items
    if (selectedItems.length === 0) {
        selectedItems = attributes.slice(0, Math.min(3, attributes.length)).map((_, idx) => idx);
    }

    const selectedNames = selectedItems.map(idx => attributes[idx].name).join(', ');
    
    const recommendation = `Style Assessment (DEMO MODE):

**Overall assessment**

You've selected some great pieces! From your wardrobe, I've chosen ${selectedNames} as the perfect combination for ${context}. ${context.toLowerCase().includes('formal') || context.toLowerCase().includes('office') ? 
    `These structured pieces work together for a polished, professional look. The darker colors give you authority while maintaining comfort and style.` :
    `These pieces create a comfortable yet stylish look that's perfect for the occasion. They work well together and can be mixed and matched easily.`}

**Why these pieces work**

For ${context}, these items are particularly well-suited because they complement each other in terms of style, color coordination, and appropriateness for the occasion. The combination creates a cohesive look that balances comfort with the expected dress code.

**Outfit combinations**

• Combine the selected items for a coordinated look
• Layer with complementary accessories
• The color palette of your chosen pieces works harmoniously
• Mix and match different pieces for variety while maintaining the overall aesthetic

**Additional styling tips**

${context.toLowerCase().includes('casual') ?
    `Keep it relaxed with rolled sleeves or untucked shirts. Add sneakers for a laid-back vibe.` :
    `Pay attention to fit and ensure everything is pressed and clean. Well-fitted clothing makes all the difference.`}

Additional items to consider:
• A quality belt that matches your shoe color
• A watch or minimal jewelry
• A complementary jacket or cardigan for layering

---
This is a demo recommendation. Add your Gemini API key to get personalized AI-powered fashion advice!
Get your free API key at: https://aistudio.google.com/app/apikey`;

    return {
        selectedItems,
        recommendation
    };
}

module.exports = {
    generateMockAttributes,
    generateMockRecommendation
};

