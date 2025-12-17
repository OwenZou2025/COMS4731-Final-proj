# FitSense Setup Quickstart

This guide helps you get FitSense running quickly using pretrained models.

## Prerequisites

- Python 3.8+ with pip
- Node.js 14+ with npm
- Git

## Step 1: Clone and Install Dependencies

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd FitSense

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
cd server
npm install
cd ..
```

## Step 2: Download Pretrained Models

**Important:** The ML server requires pretrained models. Do NOT try to train models as part of the server startup.

```bash
# Navigate to the model directory
cd custom_ml_model/model

# Run the download script
bash download_models.sh
```

This downloads:
- `best_multitask_vit.pth` (~340 MB) - **Required** for ML backend
- `checkpoint_u2net.pth` (~176 MB) - Optional, for cloth segmentation

**Manual download alternative:**
```bash
pip install gdown
gdown --id 1EFq2aVljhD_bFrLZVj_FjsKsVl-XlLz8 -O best_multitask_vit.pth
```

See `custom_ml_model/model/README.md` for more details.

## Step 3: Configure Environment

Create a `.env` file in the project root:

```bash
# Optional: Add your Gemini API key for LLM recommendations
GEMINI_API_KEY=your_api_key_here

# Optional: Add your Google API key (alternative to GEMINI_API_KEY)
GOOGLE_API_KEY=your_api_key_here

# Server ports
PORT=3000
ML_PORT=5000
```

**Note:** The system works in demo mode without API keys, using the ML backend for attribute extraction.

## Step 4: Start the Application

### Option A: Using ML Backend (Recommended)

Start the ML server (Flask):
```bash
python ml_server.py
```

In another terminal, start the Node.js server:
```bash
cd server
npm start
```

### Option B: Using Gemini API Only (No ML Server)

```bash
cd server
npm start
```

The frontend will be available at `http://localhost:3000`

## Verification

### Check ML Server Health
```bash
curl http://localhost:5000/api/ml/health
```

Expected response:
```json
{
  "status": "ok",
  "message": "ML Backend is running",
  "device": "cpu",
  "model_loaded": true
}
```

### Check Node.js Server
```bash
curl http://localhost:3000/api/health
```

## Architecture Overview

```
┌─────────────────┐
│   Frontend      │  (Vanilla JS + IndexedDB)
│  localhost:3000 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Node.js Server │  (Express + Gemini API)
│  localhost:3000 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Server     │  (Flask + PyTorch ViT)
│  localhost:5000 │
└─────────────────┘
```

## Key Files

- `ml_server.py` - Flask API that wraps ML models
- `custom_ml_model/model_architecture.py` - MultiTaskViT model definition
- `custom_ml_model/train.py` - Training script (not used by server)
- `custom_ml_model/vit_infer.py` - Single image inference
- `custom_ml_model/vit_llm_infer.py` - Batch inference with LLM
- `server/server.js` - Node.js Express server
- `frontend/` - Frontend application files

## Common Issues

**Error: "Model file not found"**
- Run `bash custom_ml_model/model/download_models.sh`
- Verify `custom_ml_model/model/best_multitask_vit.pth` exists and is ~340 MB

**Error: "gdown: command not found"**
- Install with: `pip install gdown`

**Error: "Cannot retrieve the public link"**
- Google Drive files may not be publicly accessible
- Try downloading manually or contact repository maintainer

**ML Server fails to start**
- Check Python dependencies: `pip install -r requirements.txt`
- Ensure PyTorch is installed correctly
- Check for GPU/CPU compatibility issues

**Port already in use**
- Change `PORT` or `ML_PORT` in `.env` file
- Or kill existing processes using those ports

## Training Your Own Model (Optional)

If you want to train your own model instead of using pretrained:

```bash
# Download the Kaggle dataset
kaggle datasets download -d paramaggarwal/fashion-product-images-small

# Run training
cd custom_ml_model
python train.py
```

The trained model will be saved as `best_multitask_vit.pth`. Move it to `custom_ml_model/model/` for use with the ML server.

**Note:** Training is NOT required for normal operation. The pretrained model works out of the box.

## Next Steps

1. Open `http://localhost:3000` in your browser
2. Upload clothing images to get style recommendations
3. Explore the wardrobe and history features
4. See `WARP.md` for detailed development guidance

## Support

For issues or questions:
1. Check `WARP.md` for detailed documentation
2. Review `custom_ml_model/model/README.md` for model setup
3. Check server logs for error messages
4. Verify all dependencies are installed correctly
