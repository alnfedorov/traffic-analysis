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
import cv2
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList

# Hard mapping between values in cross-road-regions.png to actual regions
CODE_TO_REGION = {
    None: "unknown",
    0:    "unknown",
    50:   "north_left",
    75:   "north_right",
    100:  "east_upper",
    125:  "east_lower",
    150:  "south_right",
    175:  "south_left",
    200:  "west_upper",
    225:  "west_lower",
    255:  "crossroad"
}
REGION_TO_CODE = {v: k for k, v in CODE_TO_REGION.items()}


class StatisticBuilder:
    def __init__(self, img_with_regions='cross-road-regions.png'):
        self.instances = {}
        if isinstance(img_with_regions, str):
            self.regions = cv2.imread(img_with_regions, cv2.IMREAD_UNCHANGED)
        else:
            self.regions = img_with_regions
        not_relevant = np.logical_and.reduce([self.regions != code for code in CODE_TO_REGION.keys()])  # Just in case
        self.regions[not_relevant] = 0
        self.last_update = None

    def _object_region(self, point):
        TO_ADJUST = 2
        x, y = point
        codes = self.regions[y-TO_ADJUST:y+TO_ADJUST, x-TO_ADJUST:x+TO_ADJUST]
        if codes.size == 0:
            return None
        values, counts = np.unique(codes.ravel(), return_counts=True)
        value = values[counts.argmax()]
        return value

    def update(self, detections: BoxList, time: float):
        self.last_update = time
        assert detections.has_field('index') and detections.mode == 'xyxy'
        for i, ind in enumerate(detections.get_field('index')):
            ind = int(ind)
            box, label, mask, score = detections.bbox[i], detections.get_field('labels')[i], \
                                      detections.get_field('mask')[i], detections.get_field('scores')[i]
            location = np.asarray([(box[0] + box[2]) / 2, box[-1]]).round().astype(np.int)  # assumed car position
            region_code = self._object_region(location)
            region = CODE_TO_REGION[region_code]  # position at the moment

            if ind in self.instances:
                self.instances[ind]['regions'].append(region)
                self.instances[ind]['labels'].append(int(label))
                self.instances[ind]['scores'].append(float(score))
                self.instances[ind]['locations'].append(location)
                self.instances[ind]['box'].append(box)
                self.instances[ind]['mask'].append(mask)
                self.instances[ind]['lost'] = self.last_update
            else:
                self.instances[ind] = {
                    "regions": [region],
                    "labels": [int(label)],
                    "scores": [float(score)],
                    "locations": [location],
                    "box": [box],
                    "mask": [mask],
                    "appeared": self.last_update,
                    "lost": self.last_update,
                }

    def finalize(self):
        results = [x for x in self.instances.values()]
        return results
