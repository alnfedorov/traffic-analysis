"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

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
import torch
import numpy as np
from numba import njit
from sklearn.utils.linear_assignment_ import linear_assignment
from maskrcnn_benchmark.structures.bounding_box import BoxList
from collections import Counter
from copy import deepcopy
from filterpy.kalman import KalmanFilter


@njit
def iou_matrix(bb_test, bb_gt):
    matrix = np.zeros((len(bb_test), len(bb_gt)), dtype=np.float32)
    bb_test_area = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    bb_gt_area = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    for i in range(len(bb_test)):
        y1 = np.maximum(bb_test[i, 0], bb_gt[:, 0])
        y2 = np.minimum(bb_test[i, 2], bb_gt[:, 2])
        x1 = np.maximum(bb_test[i, 1], bb_gt[:, 1])
        x2 = np.minimum(bb_test[i, 3], bb_gt[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = bb_test_area[i] + bb_gt_area[:] - intersection[:]
        iou = intersection / union
        matrix[i] = iou
    return matrix


def bbox_to_kalman_state(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h   # scale
    r = w / h   # ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def kalman_state_to_bbox(state):
    # state is [x, y, scale, ratio]
    w = np.sqrt(state[2] * state[3])
    h = state[2] / w
    return np.array([state[0] - w / 2, state[1] - h / 2,
                     state[0] + w / 2, state[1] + h / 2]).reshape((1, 4))


class KalmanTracker:
    NEXT_IND = 0

    def __init__(self, bbox, meta, meta_smooth_steps, img_width, img_height):
        # Filter configuration from the original SORT tracker
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_kalman_state(bbox)

        self.id = KalmanTracker.NEXT_IND
        KalmanTracker.NEXT_IND += 1

        self.history = []
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.meta_history = [meta]
        self.meta_smooth_step = meta_smooth_steps
        self.img_width, self.img_height = img_width, img_height

    def update(self, bbox, meta):
        self.time_since_update = 0
        self.hits += 1

        self.meta_history.append(meta)
        self.meta_history = self.meta_history[-self.meta_smooth_step:]

        self.kf.update(bbox_to_kalman_state(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.history.append(kalman_state_to_bbox(self.kf.x))

        self.age += 1
        self.time_since_update += 1

        box = self.history[-1][0]
        box[0], box[2] = np.clip([box[0], box[2]], 0, self.img_width)
        box[1], box[3] = np.clip([box[1], box[3]], 0, self.img_height)
        return box

    def get_state(self):
        meta = deepcopy(self.meta_history[-1])
        # smooth labels
        meta['labels'] = Counter(int(x['labels']) for x in self.meta_history).most_common(1)[0][0]
        return kalman_state_to_bbox(self.kf.x)[0], meta


def associate(detections, trackers, iou_threshold=1e-2):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=np.int), \
               np.arange(len(detections)), \
               np.empty((0, 5), dtype=np.int32)

    cost = iou_matrix(detections, trackers)
    mindices = linear_assignment(-cost)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in mindices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in mindices[:, 1]:
            unmatched_trackers.append(t)

    # filter matches with low IOU
    matches = []
    for ind in mindices:
        if cost[ind[0], ind[1]] < iou_threshold:
            unmatched_detections.append(ind[0])
            unmatched_trackers.append(ind[1])
        else:
            matches.append(ind.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []

    def update(self, dets: BoxList):
        W, H = dets.size
        assert dets.mode == 'xyxy'

        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate(dets.bbox.numpy(), trks)

        dboxes, dmeta = dets.bbox, dets.extra_fields
        per_det_meta = [{k: v[i] for k, v in dmeta.items()} for i in range(len(dets))]

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                bbox = matched[np.where(matched[:, 1] == t)[0], 0][0]
                trk.update(dboxes[bbox], per_det_meta[bbox])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanTracker(dboxes[i], per_det_meta[i], int(self.max_age * 0.5), W, H)
            self.trackers.append(trk)

        i = len(self.trackers)

        detections = {"bbox": [], "index": []}
        detections.update({k: [] for k in dets.extra_fields.keys()})

        for trk in reversed(self.trackers):
            if trk.time_since_update <= self.max_age and trk.hits >= self.min_hits:
                bbox, meta = trk.get_state()
                meta['index'] = trk.id
                detections["bbox"].append(bbox)
                for k, v in meta.items():
                    detections[k].append(v)

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(detections['bbox']) == 0:
            return None

        detections['bbox'] = torch.tensor(detections['bbox'], dtype=torch.float32)

        box_list = BoxList(detections['bbox'], (W, H))
        for k, v in detections.items():
            if k != 'bbox':
                if isinstance(v[0], torch.Tensor) and v[0].dim() != 0:
                    box_list.add_field(k, torch.cat(v))
                else:
                    box_list.add_field(k, torch.tensor(v))
        return box_list
