// Fashion Sense Server - Entry Point

require('dotenv').config({ path: require('path').join(__dirname, '../.env') });

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const FormData = require('form-data');
const fetch = require('node-fetch');

const { isDemoMode, analyzeImagesWithGemini, getRecommendationFromGemini } = require('./gemini');

const app = express();
const PORT = process.env.PORT || 3000;
const ML_BACKEND_URL = process.env.ML_BACKEND_URL || 'http://localhost:5001';

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased limit for base64 images
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(express.static(path.join(__dirname, '../frontend')));

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Log API key status on startup
console.log('Checking API Key...');
if (isDemoMode()) {
    console.log('Demo Mode: ENABLED (using mock data)');
    console.log('Reason: No valid API key detected');
    if (process.env.GEMINI_API_KEY) {
        console.log('Found placeholder value:', process.env.GEMINI_API_KEY);
    }
} else {
    console.log('API Key: VALID');
    console.log('API Key length:', process.env.GEMINI_API_KEY.length);
    console.log('First 10 chars:', process.env.GEMINI_API_KEY.substring(0, 10) + '...');
}

// API Routes

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Fashion Sense API is running' });
});

// Analyze images endpoint
app.post('/api/analyze-images', upload.array('images', 10), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No images provided' });
        }

        // Recover the original index from the filename we set on the client
        const filesInOrder = req.files
            .map((file) => {
                const [indexStr] = file.originalname.split('__');
                const originalIndex = parseInt(indexStr, 10);
                return { ...file, originalIndex: Number.isNaN(originalIndex) ? 0 : originalIndex };
            })
            .sort((a, b) => a.originalIndex - b.originalIndex);

        console.log(`Analyzing ${filesInOrder.length} images...`);

        // Use the sorted files to build buffers
        const imageBuffers = filesInOrder.map((file) => file.buffer);

        const attributes = await analyzeImagesWithGemini(imageBuffers);

        res.json({
            success: true,
            attributes,
            count: attributes.length,
        });
    } catch (error) {
        console.error('Error in analyze-images:', error);
        res.status(500).json({
            error: 'Failed to analyze images',
            message: error.message,
        });
    }
});

// ML Backend proxy endpoints
app.post('/api/ml/analyze-images', upload.array('images', 10), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: 'No images provided' });
        }

        console.log(`Forwarding ${req.files.length} images to ML backend...`);

        // Create form data to forward to ML backend
        const formData = new FormData();
        req.files.forEach((file) => {
            formData.append('images', file.buffer, {
                filename: file.originalname,
                contentType: file.mimetype
            });
        });

        // Forward to ML backend
        const mlResponse = await fetch(`${ML_BACKEND_URL}/api/ml/analyze-images`, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders()
        });

        if (!mlResponse.ok) {
            const errorText = await mlResponse.text();
            throw new Error(`ML backend error: ${errorText}`);
        }

        const mlData = await mlResponse.json();
        res.json(mlData);

    } catch (error) {
        console.error('Error in ML analyze-images:', error);
        res.status(500).json({
            error: 'Failed to analyze images with ML backend',
            message: error.message,
        });
    }
});

app.post('/api/ml/get-recommendation', async (req, res) => {
    try {
        const { attributes, context } = req.body;

        if (!attributes || !context) {
            return res.status(400).json({ error: 'Missing attributes or context' });
        }

        console.log(`Forwarding recommendation request to ML backend for context: ${context}`);

        // Forward to ML backend
        const mlResponse = await fetch(`${ML_BACKEND_URL}/api/ml/get-recommendation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ attributes, context })
        });

        if (!mlResponse.ok) {
            const errorText = await mlResponse.text();
            throw new Error(`ML backend error: ${errorText}`);
        }

        const mlData = await mlResponse.json();
        res.json(mlData);

    } catch (error) {
        console.error('Error in ML get-recommendation:', error);
        res.status(500).json({
            error: 'Failed to get recommendation from ML backend',
            message: error.message,
        });
    }
});

// ML wardrobe analysis endpoint - analyzes wardrobe items with ViT + LLM
app.post('/api/ml/analyze-wardrobe', async (req, res) => {
    try {
        const { wardrobeItems, context } = req.body;

        if (!wardrobeItems || !context) {
            return res.status(400).json({ error: 'Missing wardrobeItems or context' });
        }

        console.log(`Forwarding ${wardrobeItems.length} wardrobe items to ML backend for context: ${context}`);

        // Forward to ML backend
        const mlResponse = await fetch(`${ML_BACKEND_URL}/api/ml/analyze-wardrobe`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ wardrobeItems, context })
        });

        if (!mlResponse.ok) {
            const errorText = await mlResponse.text();
            throw new Error(`ML backend error: ${errorText}`);
        }

        const mlData = await mlResponse.json();
        res.json(mlData);

    } catch (error) {
        console.error('Error in ML analyze-wardrobe:', error);
        res.status(500).json({
            error: 'Failed to analyze wardrobe with ML backend',
            message: error.message,
        });
    }
});

// Get recommendation endpoint
app.post('/api/get-recommendation', async (req, res) => {
    try {
        const { attributes, context } = req.body;

        if (!attributes || !context) {
            return res.status(400).json({ error: 'Missing attributes or context' });
        }

        console.log(`Getting recommendation for context: ${context}`);

        // Get recommendation from Gemini
        const result = await getRecommendationFromGemini(attributes, context);

        // Handle both old format (string) and new format (object)
        const response = typeof result === 'string' 
            ? { recommendation: result, selectedItems: [] }
            : result;

        res.json({ 
            success: true,
            recommendation: response.recommendation,
            selectedItems: response.selectedItems || []
        });

    } catch (error) {
        console.error('Error in get-recommendation:', error);
        res.status(500).json({ 
            error: 'Failed to get recommendation',
            message: error.message 
        });
    }
});

// Serve frontend
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Start Server

app.listen(PORT, () => {
    console.log(`\nFashion Sense Server Running!`);
    console.log(`Server: http://localhost:${PORT}`);
    console.log(`Frontend: http://localhost:${PORT}`);
    console.log(`API Key Status: ${process.env.GEMINI_API_KEY ? 'Set' : 'Not Set'}`);
    
    if (isDemoMode()) {
        console.log(`\nRUNNING IN DEMO MODE`);
        console.log(`   The app will work with mock data for testing.`);
        console.log(`   To enable real AI analysis, add your Gemini API key to .env file.`);
        console.log(`   Get your free API key at: https://console.cloud.google.com/`);
    } else {
        console.log(`\nFull AI Mode Enabled`);
    }
    
    console.log(`\nOpen http://localhost:${PORT} in your browser to get started!\n`);
});

// Error handling
process.on('unhandledRejection', (error) => {
    console.error('Unhandled rejection:', error);
});
