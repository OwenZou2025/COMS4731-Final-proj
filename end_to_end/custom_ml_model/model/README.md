# Pretrained Model Downloads

This directory contains pretrained models for the FitSense ML backend.

## Quick Start

To download the pretrained models, run:

```bash
cd custom_ml_model/model
bash download_models.sh
```

Or manually download using `gdown`:

```bash
# Install gdown if not already installed
pip install gdown

# Download ViT model (best_multitask_vit.pth)
gdown --id 1EFq2aVljhD_bFrLZVj_FjsKsVl-XlLz8 -O best_multitask_vit.pth

# Download U2NET segmentation model (checkpoint_u2net.pth)
gdown --id 1VVBxH3pVHXy3e3zi6LmjW0kSz4I6-XFq -O checkpoint_u2net.pth
```

## Models

### 1. `best_multitask_vit.pth` (Required)
- **Purpose:** Multi-task Vision Transformer for fashion attribute prediction
- **Size:** ~340 MB
- **Predicts:** gender, articleType, baseColour, season, usage
- **Architecture:** ViT-base-patch16-224 with 5 classification heads
- **Used by:** `ml_server.py`, `vit_infer.py`, `vit_llm_infer.py`

### 2. `checkpoint_u2net.pth` (Optional)
- **Purpose:** U2NET cloth segmentation model
- **Size:** ~176 MB
- **Used by:** `cloth_segmentation.py`
- **Note:** Only needed if you want to extract clothing from images with backgrounds

## Training Your Own Model

If you want to train your own model instead of using the pretrained one:

1. Download the Kaggle Fashion Product Images dataset:
   ```bash
   kaggle datasets download -d paramaggarwal/fashion-product-images-small
   ```

2. Run the training script:
   ```bash
   python custom_ml_model/train.py
   ```

3. The trained model will be saved as `best_multitask_vit.pth` in the current directory
4. Move it to `custom_ml_model/model/` for use with the ML server

## Verification

After downloading, verify the models exist:

```bash
ls -lh custom_ml_model/model/*.pth
```

You should see:
- `best_multitask_vit.pth` (~340 MB)
- `checkpoint_u2net.pth` (~176 MB) [optional]

## Troubleshooting

**Error: "gdown: command not found"**
```bash
pip install gdown
```

**Error: "Cannot retrieve the public link"**
- Make sure the Google Drive files are publicly accessible
- Try downloading manually from the web interface
- Contact the repository maintainer for access

**Error: Model file not found when starting server**
- Ensure you've run `download_models.sh` first
- Check that `custom_ml_model/model/best_multitask_vit.pth` exists
- Verify the file is not corrupted (should be ~340 MB)
