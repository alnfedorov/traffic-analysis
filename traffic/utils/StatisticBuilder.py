import cv2
import random
import numpy as np
from matplotlib.patches import Circle
from traffic.utils.drawing import random_colors
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
REGION_TO_COLOR = {region: color for color, region in zip(random_colors(len(CODE_TO_REGION)), CODE_TO_REGION.values())}


class StatisticBuilder:
    def __init__(self, img_with_regions='cross-road-regions.png'):
        self.instances = {}
        if isinstance(img_with_regions, str):
            self.regions = cv2.imread(img_with_regions, cv2.IMREAD_UNCHANGED)
        else:
            self.regions = img_with_regions
        not_relevant = np.logical_and.reduce([self.regions != code for code in CODE_TO_REGION.keys()])  # Just in case
        self.regions[not_relevant] = 0

        self.region_colors = np.zeros((*self.regions.shape[:2], 3), dtype=np.float32)
        for ind in CODE_TO_REGION.keys():
            if ind is None or ind == 0:
                continue
            self.region_colors[self.regions == ind] = REGION_TO_COLOR[CODE_TO_REGION[ind]]
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
            box, label = detections.bbox[i], detections.get_field('labels')[i]
            location = np.asarray([(box[0] + box[2]) / 2, box[-1]]).round().astype(np.int)  # assumed car position
            region_code = self._object_region(location)
            region = CODE_TO_REGION[region_code] # position at the moment

            if ind in self.instances:
                self.instances[ind]['regions'].append(region)
                self.instances[ind]['labels'].append(int(label))
                self.instances[ind]['locations'].append(location)
                self.instances[ind]['lost'] = self.last_update
            else:
                self.instances[ind] = {
                    "regions": [region],
                    "labels": [int(label)],
                    "locations": [location],
                    "appeared": self.last_update,
                    "lost": self.last_update,
                }

    def display(self, frame, ax, regions=False, alpha=0.5):
        non_zero = (self.region_colors > 0).any(axis=-1)
        if regions:
            frame[non_zero] = np.clip(frame[non_zero] * (1-alpha)
                                      + alpha * self.region_colors[non_zero] * 255, 0, 255)
        for data in self.instances.values():
            if data['lost'] != self.last_update:
                continue
            x, y = data['locations'][-1]
            color = REGION_TO_COLOR[data['regions'][-1]]
            point = Circle((x, y), alpha=0.5, color=color, clip_on=True)
            ax.add_patch(point)

    def display_cv2(self, frame, regions=False, alpha=0.5):
        overlay = frame.copy()

        if regions:
            non_zero = (self.region_colors > 0).any(axis=-1)
            overlay[non_zero] = np.clip(self.region_colors[non_zero] * 255, 0, 255)

        for data in self.instances.values():
            if data['lost'] != self.last_update:
                continue
            x, y = data['locations'][-1]
            color = (REGION_TO_COLOR[data['regions'][-1]] * 255).astype(np.uint8)
            cv2.circle(overlay, (x, y), 4, color.tolist(), thickness=-1, lineType=cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def finalize(self):
        results = [x for x in self.instances.values()]
        return results
