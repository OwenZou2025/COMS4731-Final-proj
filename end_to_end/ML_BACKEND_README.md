# Custom ML Backend Setup

This guide explains how to set up and use the custom ML backend alongside the default Gemini backend.

## Overview

FitSense now supports two backend options:

1. **Gemini Backend** (Default): Uses Google's Gemini API for both image analysis and recommendations
2. **Custom ML Backend**: Uses your locally trained Vision Transformer (ViT) model for image analysis, combined with Gemini for recommendations

You can toggle between these backends using the button in the navigation bar.

## Architecture

### Gemini Backend (Default)
```
Frontend → Node.js Server → Gemini API
```

### Custom ML Backend
```
Frontend → Node.js Server → Python ML Server → ViT Model + Gemini API
```

## Prerequisites

Before using the Custom ML Backend, you need:

1. **Trained ViT Model**: A trained multi-task Vision Transformer model
   - Expected location: `./model/best_multitask_vit.pth`
   - Train using: `python train.py` (see main README)

2. **Python Dependencies**: Flask and ML libraries
   - Install using: `pip install -r requirements_ml_backend.txt`

3. **Node.js Dependencies**: Form data handling
   - Install using: `cd server && npm install`

## Setup Instructions

### Step 1: Install Python Dependencies

```bash
# From the project root
pip install -r requirements_ml_backend.txt
```

This installs:
- Flask & Flask-CORS (web server)
- PyTorch & Torchvision (ML framework)
- Transformers (ViT model)
- Scikit-learn (label encoding)
- Google GenAI (LLM recommendations)
- Pydantic (data validation)

### Step 2: Install Node.js Dependencies

```bash
cd server
npm install
```

This installs the new dependencies:
- `form-data`: For forwarding multipart form data to ML backend
- `node-fetch`: For making HTTP requests to ML backend

### Step 3: Ensure Model File Exists

Make sure you have a trained model at:
```
./model/best_multitask_vit.pth
```

If you don't have a trained model, train one first:
```bash
python train.py
```

## Running the Application with ML Backend

You need to run **two servers**: the Node.js server and the Python ML server.

### Terminal 1: Start the ML Backend Server

```bash
# From project root
python ml_server.py
```

This will:
- Load the ViT model into memory
- Start Flask server on port 5000
- Display: "ML Backend Server starting on port 5000..."

**Note**: The first startup may take 30-60 seconds to load the model.

### Terminal 2: Start the Node.js Server

```bash
# From project root
cd server
npm start
```

This will:
- Start Express server on port 3000
- Serve the frontend
- Proxy requests to ML backend when needed

### Step 4: Use the Application

1. Open `http://localhost:3000` in your browser
2. Click the **"Backend: Gemini"** button in the navigation bar to toggle
3. When it shows **"Backend: Custom ML"**, uploads will use your trained model
4. Toggle back to **"Gemini"** to use the default backend

## Configuration

### Environment Variables

**For ML Backend** (`.env` in project root):
```bash
# Optional: Change ML server port (default: 5000)
ML_PORT=5000

# Required for recommendations (same as Gemini backend)
GEMINI_API_KEY=your_api_key_here
```

**For Node.js Server** (`.env` in project root):
```bash
# Node server port
PORT=3000

# ML backend URL (default: http://localhost:5000)
ML_BACKEND_URL=http://localhost:5000

# Gemini API key for recommendations
GEMINI_API_KEY=your_api_key_here
```

## API Endpoints

### ML Backend (Python Flask - Port 5000)

**Health Check:**
```
GET /api/ml/health
```

**Analyze Images:**
```
POST /api/ml/analyze-images
Content-Type: multipart/form-data
Body: images (files)
```

**Get Recommendation:**
```
POST /api/ml/get-recommendation
Content-Type: application/json
Body: { "attributes": [...], "context": "..." }
```

### Node.js Server (Port 3000)

**Gemini Backend:**
- `POST /api/analyze-images`
- `POST /api/get-recommendation`

**ML Backend (proxied):**
- `POST /api/ml/analyze-images`
- `POST /api/ml/get-recommendation`

## How It Works

### When Using Custom ML Backend:

1. **User uploads images** → Frontend sends to Node.js server
2. **Node.js proxies to ML server** → `/api/ml/analyze-images`
3. **ML server processes images**:
   - Saves uploaded files temporarily
   - Runs ViT inference on each image
   - Predicts: gender, articleType, baseColour, season, usage
   - Formats results to match Gemini's output format
   - Cleans up temporary files
4. **Returns attributes to frontend** → Same format as Gemini
5. **User selects context** → Frontend requests recommendation
6. **Node.js proxies to ML server** → `/api/ml/get-recommendation`
7. **ML server generates recommendation**:
   - Groups items by type (tops/bottoms/onepieces)
   - Uses Gemini LLM for styling decision
   - Formats recommendation text
   - Returns selected item indices
8. **Frontend displays results** → Same UI as Gemini backend

### Advantages of Custom ML Backend:

✅ **Use your own trained model**: Full control over predictions  
✅ **Privacy**: Image analysis happens locally  
✅ **Customizable**: Extend with new attributes or tasks  
✅ **Cost**: Only uses Gemini API for final recommendations (not image analysis)  
✅ **Learning**: See your ML model in action  

### When to Use Each Backend:

**Use Gemini Backend when:**
- You don't have a trained model yet
- You want faster setup (no ML server needed)
- You prefer cloud-based analysis

**Use Custom ML Backend when:**
- You have a trained ViT model
- You want to use your own ML predictions
- You want to reduce API costs
- You need offline image analysis

## Troubleshooting

### ML Server Won't Start

**Error: "No module named 'flask'"**
```bash
pip install -r requirements_ml_backend.txt
```

**Error: "Model file not found"**
```bash
# Train the model first
python train.py
```

**Error: "Port 5000 already in use"**
```bash
# Change port in .env
ML_PORT=5001

# Update Node.js config
ML_BACKEND_URL=http://localhost:5001
```

### Frontend Shows Errors

**Error: "Failed to analyze images with ML backend"**
- Ensure ML server is running (`python ml_server.py`)
- Check ML server logs for errors
- Verify `ML_BACKEND_URL` is correct

**Error: "Connection refused"**
- ML server not running or wrong port
- Check firewall settings

### Slow Performance

**First inference is slow (~30-60s)**
- This is normal - model needs to load
- Subsequent requests are much faster (~1-2s per image)

**Every request is slow**
- Check if using GPU: Look for "Using device: cuda" in ML server logs
- If on CPU, consider smaller batch sizes or model optimization

## Development Notes

### Extending the ML Backend

To add new prediction tasks to the ViT model:

1. Update `train.py` to train new task heads
2. Update `ml_server.py` `format_attribute_for_frontend()` to include new predictions
3. Optionally update frontend to display new attributes

### Testing the ML Backend

**Test ML server directly:**
```bash
# Check health
curl http://localhost:5000/api/ml/health

# Test with sample image (using curl with multipart form)
curl -X POST http://localhost:5000/api/ml/analyze-images \
  -F "images=@path/to/image.jpg"
```

**Test through Node.js proxy:**
```bash
# Frontend makes requests to http://localhost:3000/api/ml/*
# Node.js forwards to http://localhost:5000/api/ml/*
```

## Performance Benchmarks

Typical performance on M1 Mac / RTX 3060:

- **Model Loading**: 30-60 seconds (one-time)
- **Single Image Analysis**: 1-2 seconds
- **5 Image Batch**: 3-5 seconds
- **Recommendation Generation**: 2-3 seconds

## Future Enhancements

Possible improvements:
- [ ] Add confidence scores from ViT model
- [ ] Support for texture/pattern prediction
- [ ] Batch processing optimization
- [ ] Model quantization for faster inference
- [ ] Docker container for ML backend
- [ ] Model versioning support

## License

Same as main project - for academic use only.
