# Copyright (c) Aleksandr Fedorov

import cv2
import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
from datetime import datetime
from traffic.utils.StatisticBuilder import StatisticBuilder
from traffic.utils.sort import Sort


parser = argparse.ArgumentParser(description="Track detected vehicles. Note, that all detections are first resized to "
                                             "fit mask crop size and then projected to the original image. Hence, "
                                             "returned coordinates are reported for the original image size. ")
parser.add_argument(
    "--predicts-file", help="path to the saved predicts", type=str, required=True
)
parser.add_argument(
    "--meta-file", help="json file with meta information about original video stream.", type=str, required=False
)
parser.add_argument(
    "--image-mask", help="path to the image mask", type=str, required=True
)
parser.add_argument(
    "--image-regions", help="path to the image mask with regions", type=str, required=True
)
parser.add_argument(
    "--max-age", help="keep tracklets that are not supported by detection at most max-age frames", type=int, default=15
)
parser.add_argument(
    "--min-hits", help="start reporting tracklets only after min-hits consequent detections", type=int, default=15
)
args = parser.parse_args()


# Get detection offsets from image mask
image_mask = cv2.imread(args.image_mask, cv2.IMREAD_UNCHANGED)
data_points = np.argwhere(image_mask > 0)
offset_y, offset_x = data_points.min(axis=0)
max_y, max_x = data_points.max(axis=0) + 1
image_mask = image_mask[offset_y:max_y, offset_x:max_x]
mask_height, mask_width = image_mask.shape[:2]


# load image with regions and predicts
IMAGE_WITH_REGIONS = cv2.imread(args.image_regions, cv2.IMREAD_UNCHANGED)
PREDICTS = torch.load(args.predicts_file)


# Read meta information and calculate approximate time delta between consequent frames
meta = args.meta_file if args.meta_file else os.path.join(os.path.dirname(args.predicts_file), "meta.json")
with open(meta, 'r') as file:
    meta = json.load(file)

stime = datetime.fromisoformat(meta['stime'])
etime = datetime.fromisoformat(meta['etime'])
per_frame_time_delta = (etime - stime) / len(PREDICTS)

tracker = Sort(max_age=args.max_age, min_hits=args.min_hits)
statistic_builder = StatisticBuilder(IMAGE_WITH_REGIONS)
time = stime

for d in tqdm(PREDICTS):
    d = d.convert('xyxy').resize((mask_width, mask_height))

    d.bbox[:, 0] += offset_x
    d.bbox[:, 2] += offset_x
    d.bbox[:, 1] += offset_y
    d.bbox[:, 3] += offset_y
    d.size = (IMAGE_WITH_REGIONS.shape[1], IMAGE_WITH_REGIONS.shape[0])

    detections = tracker.update(d)

    if detections is not None:
        statistic_builder.update(detections, time)

    time = time + per_frame_time_delta

results = statistic_builder.finalize()
results = [x for x in results if set(x['regions']) != set(['unknown'])]

# Merge all detections within short temporal and spatial window
merged = []
available = np.ones(len(results), dtype=np.bool)
for ind, root in tqdm(enumerate(results)):
    if not available[ind]:
        continue

    appeared, lost = root['appeared'], root['lost']
    appeared_loc, lost_loc = root['locations'][0], root['locations'][-1]

    r = deepcopy(root)

    for ind_merge, to_merge in enumerate(results[ind+1:]):
        ind_merge = ind + 1 + ind_merge
        if not available[ind_merge]:
            continue

        # merge to end
        if r['lost'] < to_merge['appeared'] and (to_merge['appeared'] - r['lost']).total_seconds() < 3:
            if np.linalg.norm(lost_loc - to_merge['locations'][0]) < 20:
                available[ind_merge] = False
                r['locations'] += to_merge['locations']
                r['regions'] += to_merge['regions']
                r['labels'] += to_merge['labels']
                r['lost'] = to_merge['lost']

        # merge at the beginning
        if r['appeared'] > to_merge['lost'] and (r['appeared'] - to_merge['lost']).total_seconds() < 3:
            if np.linalg.norm(appeared_loc - to_merge['locations'][-1]) < 20:
                available[ind_merge] = False
                r['locations'] = to_merge['locations'] + r['locations']
                r['regions'] = to_merge['regions'] + r['regions']
                r['labels'] = to_merge['labels'] + r['labels']
                r['appeared'] = to_merge['appeared']
    merged.append(r)
results = merged


# Map image directions to the real one. Also take into account possible bugs in tracking
DIRECTIONS = {
    "north_left": {
        "west_upper":  "W-S",
        "east_lower":  "W-N",
        "east_upper":  "W-N",  # tracking bug
        "south_left":  "W-E",
        "north_right": "W-W"
    },
    "south_right": {
        "west_upper":  "E-S",
        "west_lower":  "E-S",  # tracking bug
        "east_lower":  "E-N",
        "east_upper":  "E-N",  # tracking bug
        "north_right": "E-W",
        "south_left":  "E-E"
    },
    "west_lower": {
        "north_right": "S-W",
        "east_upper":  "S-N",  # tracking bug
        "east_lower":  "S-N",
        "south_left":  "S-E",
        "west_upper":  "S-S"
    },
    "east_upper": {
        "north_right": "N-W",
        "west_upper":  "N-S",
        "west_lower":  "N-S",  # tracking bug
        "south_left":  "N-E",
        "east_lower":  "N-N"
    },
    "east_lower": {            # tracking bug
        "north_right": "N-W",
        "west_upper":  "N-S",
        "south_left":  "N-E"
    },
    "west_upper": {
        "north_right": "S-W",  # tracking bug
    }
}

detections = []
for r in tqdm(results):
    regions = r['regions']
    unique = set(regions).difference({'unknown'})
    if len(unique) < 3 or 'crossroad' not in unique:
        continue

    regions = np.asarray(regions)

    # 1. Crossroad in the middle
    split = np.where(regions == 'crossroad')[0]
    split = np.clip(split.mean().round().astype(np.int), 0, len(regions) - 1)

    # 2. Split time for before and after
    before = regions[:split]
    after = regions[split:]
    before, after = before[(before != "crossroad") & (before != "unknown")], \
                    after[(after != "crossroad") & (after != "unknown")]

    # 3. Most common directions before and after crossroad
    approximate_direction = lambda x: Counter(x).most_common(1)[0][0] if x.size != 0 else None
    arrived_from = approximate_direction(before)
    departured_to = approximate_direction(after)
    if arrived_from is None or departured_to is None:
        continue

    if arrived_from not in DIRECTIONS:
        continue
    if departured_to not in DIRECTIONS[arrived_from]:
        continue
    label = Counter(r['labels']).most_common(1)[0][0]
    d = DIRECTIONS[arrived_from][departured_to]
    detections.append(
        {
            "appeared": r['appeared'].time().isoformat(),
            "lost": r['lost'].time().isoformat(),
            "direction": d,
            "label": label
        }
    )


tracklets = os.path.join(os.path.dirname(args.predicts_file), "tracklets.json")
with open(tracklets, 'w') as tracklets:
    json.dump(detections, tracklets)