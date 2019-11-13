# Copyright (c) Aleksandr Fedorov

import sys,os
sys.path.append(os.getcwd())

import os
import cv2
import torch
import argparse
import time
import itertools
import numpy as np
from traffic.utils.ModelWrapper import ModelWrapper


parser = argparse.ArgumentParser(description="Detect vehicles in the video stream. "
                                             "Video stream should NOT be cropped or masked in advance")
parser.add_argument(
    "--config-file", help="path to config file", type=str, required=True
)
parser.add_argument(
    "--video-file", help="path to the requested .mp4 video", type=str, required=True
)
parser.add_argument(
    "--batch-size", default=1, help="number of frames in the batch", type=int
)
parser.add_argument(
    "--backup-frames", default=500, help="backup progress every k frames", type=int
)
parser.add_argument(
    "--image-mask", help="path to the image mask", type=str, required=True
)
parser.add_argument(
    "--save-to", help="path where to save results", type=str, required=True
)
args = parser.parse_args()


BATCH_SIZE, BACKUP_EVERY_FRAMES = args.batch_size, args.backup_frames
VIDEO_FILE, CONFIG_FILE = args.video_file, args.config_file
assert os.path.exists(VIDEO_FILE) and os.path.exists(CONFIG_FILE) and os.path.exists(args.image_mask), \
    "Requested files don't exist"


# Initialize cache and load predicts if any
TARGET_FILE = args.save_to
CACHE = []
SKIP_FRAMES = 0
if os.path.exists(TARGET_FILE):
    PREDICTS = torch.load(TARGET_FILE)
    SKIP_FRAMES += len(PREDICTS)
else:
    PREDICTS = []


# load model and image mask
model = ModelWrapper(CONFIG_FILE, confidence_threshold=0.3, mask_threshold=0.75, mask_on=True)

IMAGE_MASK = cv2.imread(args.image_mask, cv2.IMREAD_UNCHANGED)
FRAME_H, FRAME_W = IMAGE_MASK.shape[:2]
data_points = np.argwhere(IMAGE_MASK > 0)
min_y, min_x = data_points.min(axis=0)
max_y, max_x = data_points.max(axis=0) + 1

IMAGE_MASK = IMAGE_MASK[min_y:max_y, min_x:max_x, None]
IMAGE_MASK[IMAGE_MASK > 0] = 1

def predict(cache):
    detections = model.predict(cache, project_mask=False)
    for i in range(len(detections)):
        detections[i] = detections[i].convert('xyxy').clip_to_image(remove_empty=True)
        detections[i].size = (FRAME_W, FRAME_H)
        detections[i].bbox.add_(torch.tensor([min_x, min_y, min_x, min_y], dtype=torch.float32))
    return detections

# open video stream and skip processed frames
stream = cv2.VideoCapture(VIDEO_FILE)
for i in range(SKIP_FRAMES):
    print(f"skipping frame {i+1}, already processed")
    ret, frame = stream.read()
    assert ret


spinner = itertools.cycle(['-', '/', '|', '\\'])
ret = True
start = time.time()

NEXT_BACKUP = len(PREDICTS) + BACKUP_EVERY_FRAMES

while stream.isOpened() and ret:
    ret, frame = stream.read()

    if not ret:
        break

    if ret and (frame.ndim != 3 or frame.shape[-1] != 3):
        print("WARNING: Corrupted frame. Inserting None as a result")
        # Drop current cache
        PREDICTS += predict(np.asarray(CACHE))
        CACHE = []
        # Insert None
        PREDICTS.append(None)

        print(f"{next(spinner)} frames finished {len(PREDICTS)}",
              flush=True, end='\r')
        continue

    frame = frame[min_y:max_y, min_x:max_x]
    assert frame.shape[:2] == IMAGE_MASK.shape[:2] and frame.ndim == IMAGE_MASK.ndim
    CACHE.append(frame * IMAGE_MASK)

    if len(CACHE) % BATCH_SIZE == 0 and len(CACHE) != 0:
        PREDICTS += predict(np.asarray(CACHE))

        if len(PREDICTS) >= NEXT_BACKUP:
            speed = (len(PREDICTS) - SKIP_FRAMES) / (time.time() - start)
            print(f"Backup, {speed} iter/sec...")
            torch.save(PREDICTS, TARGET_FILE)
            NEXT_BACKUP += BACKUP_EVERY_FRAMES

        CACHE = []
        print(f"{next(spinner)} frames finished {len(PREDICTS)}",
              flush=True, end='\r')


if len(CACHE) != 0:
    PREDICTS += predict(np.asarray(CACHE))


print(f"\nTotal images processed {len(PREDICTS)}. Saving results..")
torch.save(PREDICTS, TARGET_FILE)
