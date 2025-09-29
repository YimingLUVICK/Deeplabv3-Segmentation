from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import json
import random

def color_map():
    color2id = {}
    with open("color2id.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    for k, v in raw.items():
        color = tuple([int(p) for p in k[1:-1].split(',')])
        color2id[color] = int(v)
    return color2id

def invert_color_map(color2id: dict, num_classes=15):
    id2color = np.zeros((num_classes, 3), dtype=np.uint8)
    for (r, g, b), cid in color2id.items():
        if 0 <= cid < num_classes:
            id2color[cid] = np.array([r, g, b], dtype=np.uint8)
    return id2color

color2id = color_map()
id2color = invert_color_map(color2id)

def resize(image: Image.Image, label = False, target_size=(256, 256)):
    orig_w, orig_h = image.size
    targ_w, targ_h = target_size

    scale = min(targ_w / orig_w, targ_h / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    pad_left = (targ_w - new_w) // 2
    pad_right = targ_w - new_w - pad_left
    pad_top = (targ_h - new_h) // 2
    pad_bottom = targ_h - new_h - pad_top

    if not label:
        new_image = image.resize((new_w, new_h), Image.BILINEAR)
        new_image = ImageOps.expand(new_image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=128)
    else:
        new_image = image.resize((new_w, new_h), Image.NEAREST)
        new_image = ImageOps.expand(new_image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)
        
    meta = (scale, pad_left, pad_top, new_w, new_h, orig_w, orig_h)
    
    return new_image, meta

def unresize(id_mask_256: np.ndarray, meta):
    scale, pad_left, pad_top, new_w, new_h, orig_w, orig_h = meta
    crop = id_mask_256[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    pil_mask = Image.fromarray(crop.astype(np.uint8), mode="L")
    pil_mask = pil_mask.resize((orig_w, orig_h), Image.NEAREST)
    
    return np.array(pil_mask, dtype=np.uint8)

def rgb_to_idmask(rgb_img: np.ndarray, color2id: dict):
    H, W, _ = rgb_img.shape
    mask = np.full((H, W), 255, dtype=np.int64)

    code_img = (rgb_img[:,:,0].astype(np.int64) << 16) + \
               (rgb_img[:,:,1].astype(np.int64) << 8) + \
               rgb_img[:,:,2].astype(np.int64)

    code2id = {}
    for (r,g,b), cid in color2id.items():
        code = (int(r)<<16) + (int(g)<<8) + int(b)
        code2id[code] = cid

    for code, cid in code2id.items():
        mask[code_img == code] = cid

    return mask

def idmask_to_rgb(id_mask: np.ndarray, id2color: np.ndarray):
    h, w = id_mask.shape
    max_id = id2color.shape[0] - 1
    safe = np.clip(id_mask, 0, max_id)
    return id2color[safe]

## Here are the function for data augmentation 

def random_h_flap(image: Image.Image, label: Image.Image, p: float = 0.5):
    if random.random() < p:
        image = ImageOps.mirror(image)
        label = ImageOps.mirror(label)
    return image, label

def random_rotate(image: Image.Image, label: Image.Image, p: float = 0.5):
    if random.random() < p:
        angle = random.uniform(-10, 10)
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=128)
        label = label.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=255)
    return image, label

def random_gaussian_blur(image: Image.Image, label: Image.Image, radius_range=(0.1, 2.0), p: float = 0.3):
    if random.random() < p:
        radius = random.uniform(*radius_range)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image, label
        