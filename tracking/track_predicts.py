"""
    Copyright (c) Aleksandr Fedorov canxes@mail.ru
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from StatisticBuilder import StatisticBuilder
from sort import Sort

parser = argparse.ArgumentParser(description="Track detected vehicles.")
parser.add_argument(
    "--predicts-file", help="path to the saved predicts", type=str, required=True
)
parser.add_argument(
    "--image-regions", help="path to the image mask with regions", type=str, required=True
)
parser.add_argument(
    "--max-age", help="keep tracklets that are not supported by detections at most max-age frames", type=int, default=15
)
parser.add_argument(
    "--min-hits", help="start reporting tracklets only after min-hits consequent detections", type=int, default=15
)
parser.add_argument(
    "--save-to", help="path where to save results", type=str, required=True
)
args = parser.parse_args()

assert os.path.exists(args.image_regions) and os.path.exists(args.predicts_file), "Files don't exist"
TARGET_FILE = args.save_to
# load image with regions and predicts
IMAGE_WITH_REGIONS = cv2.imread(args.image_regions, cv2.IMREAD_UNCHANGED)
PREDICTS = torch.load(args.predicts_file)


# track objects
PER_FRAME_DELTA = 1

tracker = Sort(max_age=args.max_age, min_hits=args.min_hits)
statistic_builder = StatisticBuilder(IMAGE_WITH_REGIONS)
time = 0

for d in tqdm(PREDICTS):
    detections = tracker.update(d)

    if detections is not None:
        statistic_builder.update(detections, time)

    time += PER_FRAME_DELTA

results = statistic_builder.finalize()

# TODO add proper interpolation between merged objects
# Merge all detections within short temporal and spatial window
# PIXELS_DISTANCE_THR = 20
# FRAMES_DISTANCE_THR = 50
# merged = []
# available = np.ones(len(results), dtype=np.bool)
# assert all(r['lost'] - r['appeared'] + 1 == len(r['locations']) for r in results)
# for ind, root in tqdm(enumerate(results)):
#     if not available[ind]:
#         continue
#
#     appeared, lost = root['appeared'], root['lost']
#     appeared_loc, lost_loc = root['locations'][0], root['locations'][-1]
#
#     r = root
#
#     for ind_merge, to_merge in enumerate(results[ind+1:]):
#         ind_merge = ind + 1 + ind_merge
#         if not available[ind_merge]:
#             continue
#
#         # merge to end
#         if r['lost'] < to_merge['appeared'] and abs(to_merge['appeared'] - r['lost']) < FRAMES_DISTANCE_THR:
#             if np.linalg.norm(lost_loc - to_merge['locations'][0]) < PIXELS_DISTANCE_THR:
#                 available[ind_merge] = False
#                 appeared, lost = r['appeared'], to_merge['lost']
#                 for k in ['regions', 'labels', 'locations']:
#                     r[k][:-1] += to_merge[k]
#                 r['appeared'], r['lost'] = appeared, lost
#                 appeared_loc, lost_loc = r['locations'][0], r['locations'][-1]
#                 assert r['lost'] - r['appeared'] + 1 == len(r['locations'])
#
#         # merge at the beginning
#         if r['appeared'] > to_merge['lost'] and abs(r['appeared'] - to_merge['lost']) < FRAMES_DISTANCE_THR:
#             if np.linalg.norm(appeared_loc - to_merge['locations'][-1]) < PIXELS_DISTANCE_THR:
#                 available[ind_merge] = False
#                 appeared, lost = to_merge['appeared'], r['lost']
#                 for k in r.keys():
#                     r[k] = to_merge[k][:-1] + r[k]
#                 r['appeared'], r['lost'] = appeared, lost
#                 appeared_loc, lost_loc = r['locations'][0], r['locations'][-1]
#                 assert r['lost'] - r['appeared'] + 1 == len(r['locations'])
#
#     merged.append(r)
# results = merged


# Map image directions to the real one. Also take into account possible bugs in tracking
DIRECTIONS = {
    "north_left": {
        "west_upper":  "E-N",
        "east_lower":  "E-S",
        "east_upper":  "E-S",  # tracking bug
        "south_left":  "E-W",
        "north_right": "E-E"
    },
    "south_right": {
        "west_upper":  "W-N",
        "west_lower":  "W-N",  # tracking bug
        "east_lower":  "W-S",
        "east_upper":  "W-S",  # tracking bug
        "north_right": "W-E",
        "south_left":  "W-W"
    },
    "west_lower": {
        "north_right": "N-E",
        "east_upper":  "N-S",  # tracking bug
        "east_lower":  "N-S",
        "south_left":  "N-W",
        "west_upper":  "N-N"
    },
    "east_upper": {
        "north_right": "S-E",
        "west_upper":  "S-N",
        "west_lower":  "S-N",  # tracking bug
        "south_left":  "S-W",
        "east_lower":  "S-S"
    },
    "east_lower": {            # tracking bug
        "north_right": "S-E",
        "west_upper":  "S-N",
        "south_left":  "S-W"
    },
    "west_upper": {
        "north_right": "N-E",  # tracking bug
    }
}

for r in tqdm(results):
    label = Counter(r['labels']).most_common(1)[0][0]
    regions = r['regions']

    r.pop('labels')
    r.pop('regions')
    r.pop('locations')

    r['direction'] = 'unknown'
    r['label'] = label

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
    d = DIRECTIONS[arrived_from][departured_to]

    r['direction'] = d


torch.save(results, TARGET_FILE)
