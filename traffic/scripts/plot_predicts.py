import torch
import argparse
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from traffic.utils.drawing import random_colors, display_instances
from traffic.utils.ModelWrapper import ModelWrapper
from traffic.utils.sort import Sort
from traffic.utils.StatisticBuilder import StatisticBuilder
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description="Render pretty images with detection results")
parser.add_argument(
    "--config-file", metavar="FILE", help="path to the model config file", type=str, required=True
)
parser.add_argument(
    "--video-file", metavar="FILE", help="path to the raw .mp4 video (no preprocessing)", type=str, required=True
)
parser.add_argument(
    "--save-to", metavar="FOLDER", help="target folder for resilts", type=str, required=True
)
parser.add_argument(
    "--image-mask", metavar="FILE", help="mask for model input images", type=str, required=True
)
parser.add_argument(
    "--image-regions", metavar="FILE", help="image with road regions", type=str, required=True
)
args = parser.parse_args()


CLASS_NAMES = {
     1: "car",
     2: "trolleybus",
     3: "tram",
     4: "truck",
     5: "bus",
     6: "van"
}

image_mask = cv2.imread(args.image_mask, cv2.IMREAD_UNCHANGED)
image_mask[image_mask > 0] = 1
data_points = np.argwhere(image_mask)
min_y, min_x = data_points.min(axis=0)
max_y, max_x = data_points.max(axis=0) + 1
image_mask = image_mask[min_y:max_y, min_x:max_x]
assert image_mask.shape == (769, 1920)
image_mask[image_mask > 0] = 1
image_mask = image_mask[:, :, None]


video_file = args.video_file
folder = args.save_to
os.makedirs(folder, exist_ok=True)


tracker = Sort(max_age=15, min_hits=15)
statistic_builder = StatisticBuilder(args.image_regions)
model = ModelWrapper(args.config_file, confidence_threshold=0.3, mask_threshold=0.5)
stream = cv2.VideoCapture(video_file)

# id -> color
colors_generator = defaultdict(lambda *args: random_colors(N=500)[0])   # Not well suitable, but works

DPI = plt.rcParams['figure.dpi']
FRAME_ID = 0

plt.figure(figsize=(1920 / DPI, 1080 / DPI), dpi=DPI)
plt.subplots_adjust(bottom=0, right=1, top=1, left=0)

while stream.isOpened():
    print("Current frame ", FRAME_ID)
    ret, frame = stream.read()

    assert frame.ndim == 3 and frame.shape[-1] == 3
    H, W = frame.shape[:2]

    if not ret:
        print("Failed to read frame. Terminating further processing")
        break

    det_frame = frame[min_y:max_y, min_x:max_x] * image_mask
    detections = model.predict([det_frame], project_mask=False)[0]
    detections = detections.convert('xyxy').clip_to_image(remove_empty=True)

    detections.size = (W, H)
    detections.bbox.add_(torch.tensor([min_x, min_y, min_x, min_y], dtype=torch.float32))
    detections = tracker.update(detections)

    if detections is None:
        continue

    if detections.get_field('mask').dim() != 4:
        detections.get_field('mask').data = detections.get_field('mask').unsqueeze(1)

    detections = detections.resize((W, H)).convert('xyxy').clip_to_image(remove_empty=True)
    statistic_builder.update(detections, FRAME_ID)
    detections = model.project_masks(detections)

    colors = [colors_generator[int(ind)] for ind in detections.get_field("index")]

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # statistic_builder.display(frame, ax, regions=False, alpha=0.5)
    # BGR -> RGB
    frame = frame[:, :, [2, 1, 0]]

    display_instances(frame, detections, CLASS_NAMES, ax, colors=colors)
    plt.savefig(os.path.join(folder, f'image-{FRAME_ID}.png'), bbox_inches='tight',
                pad_inches=0, dpi=DPI)


    FRAME_ID += 1
    plt.clf()
