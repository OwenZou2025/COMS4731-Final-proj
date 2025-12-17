import os
from PIL import Image
import cv2
import gdown
import argparse
import numpy as np
from networks import U2NET
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"

def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)



def generate_mask(input_image, net, palette, img_name, device = 'cpu'):

    #img = Image.open(input_image).convert('RGB')
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    alpha_out_dir = "./input/masks"
    cloth_seg_out_dir = "./input/masks"

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    for cls in range(1, 4):
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_mask_img.save(os.path.join(alpha_out_dir, img_name + f'{cls}.png'))

    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    cloth_seg.save(os.path.join(cloth_seg_out_dir, img_name + '_final_seg.png'))
    return cloth_seg

def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net

def combine_masks(mask_paths):
    """Combine multiple mask files into a single unified mask.
    
    Args:
        mask_paths: List of paths to mask files
    
    Returns:
        Combined mask as numpy array, or None if no masks
    """
    if not mask_paths:
        return None
    
    import cv2
    combined_mask = None
    
    for mask_path in mask_paths:
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        if combined_mask is None:
            combined_mask = mask
        else:
            # Combine using logical OR to include all clothing regions
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask

def segmentation(mask_dir, original_dir, output_dir, white_bg_only=True):
    """Apply mask to original image and create segmented clothing.
    
    Args:
        mask_dir: Path to the mask image
        original_dir: Path to the original image
        output_dir: Directory to save output
        white_bg_only: If True, only save white background version (default)
    """
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(original_dir)

    base = os.path.splitext(os.path.basename(mask_dir))[0]
    output_prefix = os.path.join(output_dir, base)

    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    cloth_extracted = cv2.bitwise_and(original, original, mask=binary_mask)
    inverse_mask = cv2.bitwise_not(binary_mask)
    white_background = np.ones_like(original) * 255
    background = cv2.bitwise_and(white_background, white_background, mask=inverse_mask)
    cloth_with_white_bg = cv2.add(cloth_extracted, background)

    # Always save white background version
    cv2.imwrite(output_prefix + '_white_bg.png', cloth_with_white_bg)
    
    # Optionally save other versions
    if not white_bg_only:
        original_rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        original_rgba[:, :, 3] = binary_mask
        cv2.imwrite(output_prefix + '_black_bg.png', cloth_extracted)
        cv2.imwrite(output_prefix + '_transparent.png', original_rgba)
    
    return output_prefix + '_white_bg.png'


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_seg_model("./model/checkpoint_u2net.pth", device=device)
    palette = get_palette(4)
    input_dir = "./input/raw_images"
    output_dir = "./output/segmented_clothes"
    valid_formats = {".jpg", ".jpeg", ".png"}
    for filename in os.listdir(input_dir):
        form = os.path.splitext(filename)[1].lower()
        img_name = os.path.splitext(filename)[0]
        if form not in valid_formats:
            continue
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")

        cloth_seg = generate_mask(img, net=model, palette=palette, img_name=img_name, device=device)

        mask_dir = "./input/masks"
        original_dir = img_path

        for cls in [1, 2, 3]:
            mask_path = os.path.join(mask_dir, f"{img_name}{cls}.png")
            if os.path.exists(mask_path):
                segmentation(mask_path, original_dir, output_dir)