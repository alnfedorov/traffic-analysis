import cv2
import numpy as np
import random
import colorsys
from matplotlib import patches


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return np.asarray(colors).astype(np.float32)


def apply_mask(image, mask, color, alpha: float=0.5):
    index = (mask > 0).nonzero()
    image[index] = image[index] * (1 - alpha) + alpha * color * 255
    return image


def display_instances_cv2(image, boxlist, class_names, colors=None):
    N = len(boxlist)
    if colors is None:
        colors = random_colors(N)
    assert len(colors) == N

    height, width = image.shape[:2]
    boxlist = boxlist.resize((width, height)).convert('xyxy')

    image = image.astype(np.uint8)
    overlay = image.copy()
    for i in range(N):
        color = (colors[i] * 255).astype(np.uint8).tolist()
        box, score, label = boxlist.bbox[i], boxlist.get_field('scores')[i], boxlist.get_field('labels')[i]
        box, score, label = box.cpu().numpy(), float(score), int(label)

        if not np.any(box):
            continue

        x0, y0, x1, y1 = box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color=color, thickness=2, lineType=cv2.LINE_AA)

        # Label
        caption = "{} {:.3f}".format(class_names[label], score) if score else label
        cv2.putText(overlay, caption, (int(x0), int(y0+8)), cv2.QT_FONT_NORMAL, 0.5,
                    color=color, thickness=2, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.5, image, 1 - 0.5, 0)


def display_instances(image, boxlist, class_names, ax, colors=None, draw_masks=True):
    N = len(boxlist)
    if colors is None:
        colors = random_colors(N)
    assert len(colors) == N

    height, width = image.shape[:2]
    boxlist = boxlist.resize((width, height)).convert('xywh')

    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        color = colors[i]
        box, score, label = boxlist.bbox[i], boxlist.get_field('scores')[i], boxlist.get_field('labels')[i]
        box, score, label = box.cpu().numpy(), float(score), int(label)

        if not np.any(box):
            continue

        x, y, w, h = box
        # x, y, w, h = max(0, x-1), max(0, y-1), max(0, w-1), max(0, h-1)
        p = patches.Rectangle((x, y), w, h, linewidth=2,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=color, facecolor='none', clip_on=True)
        ax.add_patch(p)

        # Label
        caption = "{} {:.3f}".format(class_names[label], score) if score else label
        ax.text(x, max(0, y + 8), caption, color='w', size=8, backgroundcolor="none", clip_on=True)

        # Mask
        if draw_masks:
            mask = boxlist.get_field('mask')[i].cpu().numpy()
            mask = mask.reshape(height, width)
            masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask

    ax.imshow(masked_image.astype(np.uint8))
    return masked_image