FILE_ID_1="1EFq2aVljhD_bFrLZVj_FjsKsVl-XlLz8"
OUTPUT_PATH_1="best_multitask_vit.pth"

FILE_ID_2="1VVBxH3pVHXy3e3zi6LmjW0kSz4I6-XFq"
OUTPUT_PATH_2="checkpoint_u2net.pth"

gdown --id $FILE_ID_1 -O $OUTPUT_PATH_1
echo "Download complete! Saved to $OUTPUT_PATH_1"

gdown --id $FILE_ID_2 -O $OUTPUT_PATH_2
echo "Download complete! Saved to $OUTPUT_PATH_2"