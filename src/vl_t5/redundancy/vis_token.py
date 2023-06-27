import os
import cv2
import random
import colorsys
import numpy as np
from PIL import Image
from skimage.measure import find_contours
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import torch
import torch.nn as nn
from torchvision import transforms

matplotlib.use('TkAgg')


def tensor2im_norm(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.detach().cpu().float().numpy()
    image_numpy = image_numpy - np.min(image_numpy)
    image_numpy = image_numpy / np.max(image_numpy) * 255
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def vis_hm(token_embed, image=None, is_save=False, is_show=False, img_path=None):
    if img_path is not None:
        # normalize = transforms.Normalize(
        #     (0.48145466, 0.4578275, 0.40821073),
        #     (0.26862954, 0.26130258, 0.27577711))
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            # normalize,
        ])
        
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
    
    # :1 保留维度
    token_att = token_embed[..., :1, :] @ token_embed.transpose(-2, -1)
    token_att = torch.sigmoid(token_att.squeeze(-2) * 0.125)  # [bs, n_seq]
    
    token_d = int(np.sqrt(token_att.shape[1] - 1))
    soft_map = token_att[:, 1:]
    hm = soft_map.view(soft_map.size(0), token_d, token_d)
    hm = hm.squeeze()
    hm = torch.clamp(hm, 0.48, 1)
    
    vis = tensor2im_norm(hm.view(token_d, token_d, 1))
    h, w = image.size(1), image.size(2)
    heatmap = cv2.applyColorMap(cv2.resize(vis, (w, h)), cv2.COLORMAP_JET)
    
    image = image * 255
    original = image.transpose(0, 1).transpose(1, 2).cpu().numpy().astype(
        np.uint8)
    original = original[:, :, ::-1]
    heatmap = heatmap.astype(np.uint8)
    result = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    
    if is_save:
        cv2.imwrite(os.path.join(
            'result', 'interpret-result.png'), result)
    if is_show:
        im = Image.fromarray(result)
        im.show()
    
    # return result


# ---------------vis activated token position---------------
def vis_activated_token(image=None, keep_idx=None,
                        is_save=False, is_show=False, img_path=None):
    if img_path is not None:
        transform = transforms.Compose([
            transforms.Resize((600, 600), interpolation=3),
            transforms.ToTensor(),
        ])
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
    image = image * 255  # !!!
    original = image.transpose(0, 1).transpose(1, 2).cpu().numpy().astype(
        np.uint8)
    original = original[:, :, ::-1]
    
    rows = 30
    cols = 30
    
    chunks = []
    idx = 0
    for row_img in np.array_split(original, rows, axis=0):
        for chunk in np.array_split(row_img, cols, axis=1):
            if idx in keep_idx:
                mask_area = cv2.rectangle(
                    np.zeros((20, 20, 3)), (0, 0), (20, 20),
                    color=(255, 0, 255), thickness=-1)
                mask_area = cv2.rectangle(
                    mask_area, (0, 0), (20, 20),
                    color=(255, 255, 255), thickness=1)
            else:
                mask_area = cv2.rectangle(
                    np.zeros((20, 20, 3)), (0, 0), (20, 20),
                    color=(255, 255, 255), thickness=1)
            chunk = cv2.addWeighted(
                chunk, 1, mask_area.astype(np.uint8), 0.5, 0)
            chunks.append(chunk)
            idx += 1
    img = np.stack(chunks)
    img = img.reshape(rows, cols, 20, 20, 3)
    img = [np.concatenate(col_imgs, axis=1) for col_imgs in img]
    img = np.concatenate(img, axis=0)
    
    if is_save:
        cv2.imwrite(os.path.join(
            'result', 'interpret-result.png'), img)
    if is_show:
        im = Image.fromarray(img)
        im.show()


def show_assignments(a, b, P):
    norm_P = P / P.max()
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            plt.arrow(a[i, 0], a[i, 1], b[j, 0] - a[i, 0], b[j, 1] - a[i, 1],
                      alpha=norm_P[i, j].item())
    plt.title('Assignments')
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(b[:, 0], b[:, 1])
    plt.axis('off')


# *****************************************************************************
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * \
                         color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False,
                      contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()
    
    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)
    
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(os.path.join("exp_analysis/vis_result", fname))
    # print(f"{fname} saved.")
    return



