# COMS4731-Final-proj
## 1. Install Dependencies
Install all required libraries using:

```bash
pip install -r requirements.txt
```
## 2. Download models
The pretrained or fine-tuned model has been uploaded to Google Drive. Download them by:

```bash
bash ./model/download_model.sh
```

## 3. Cloth segmentation
The initial input images are in the path ./input/raw_images.
```bash
python cloth_segmentation.py
```
The code will automatically transfer all images in the raw_images folder to masks in the masks folder and segment clothes in the output folder.
