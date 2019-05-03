# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


class HorizontalSegmentMedianBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if self.prob == 0:
            return image, target
        if random.random() < self.prob:
            image, mode = np.asarray(image), image.mode
            if random.random() < 0.75:
                image = image.copy()
                height, width = image.shape[:2]
                y = int(height * np.random.uniform(0.25, 0.7))
                h = int(height * np.random.uniform(0.15, 0.3))
                # Same as on the video
                ksize = random.choice([5, 7, 9])
                ksize = min(ksize, h)
                image[y:y+h, :] = cv2.blur(image[y:y+h,:], (ksize, ksize))
            else:
                image = cv2.blur(image, (5, 5)) # Blur everything
            image = Image.fromarray(image, mode)
        return image, target


class ContrastAdjust(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if self.prob == 0:
            return image, target
        if random.random() < self.prob:
            factor = np.random.uniform(0.25, 0.75)
            image = F.adjust_contrast(image, factor)
        return image, target


class BrightnessAdjust(object):
    def __init__(self, prob=0.5, factor=(0.25, 0.75)):
        self.prob = prob
        self.factor = factor

    def __call__(self, image, target):
        if self.prob == 0:
            return image, target
        if random.random() < self.prob:
            factor = np.random.uniform(self.factor)
            image = F.adjust_brightness(image, factor)
        return image, target


class InstanceMasking(object):
    def __init__(self, prob=0.5, min_area_to_mask=40, max_area_hidden=0.45):
        self.prob = prob
        self.max_area_hidden = max_area_hidden
        self.min_area_to_mask = min_area_to_mask

    def _line_mask(self, mask, x, y, w, h):
        if random.random() < 0.5:   # horizontal line
            line_y, line_h = np.random.randint(y, y+h), np.random.randint(0, int(h*self.max_area_hidden))
            line_x, line_w = x, w
        else:                       # vertical line
            line_y, line_h = y, h
            line_x, line_w = np.random.randint(x, x+w), np.random.randint(0, int(w*self.max_area_hidden))

        x1, y1 = line_x, line_y
        x2, y2 = min(x + w, x1 + line_w), min(y+h, y1+line_y)
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        hidden_area = (x2 - x1) * (y2 - y1)
        if hidden_area / (h*w) > self.max_area_hidden or hidden_area < 1:
            return
        mask[y1:y2, x1:x2] += 1

    def _ellipse_mask(self, mask, x, y, w, h):
        max_area = int(w * h * self.max_area_hidden)
        if max_area < 10:
            return
        area = max_area / np.pi
        a = random.random() * area
        b = area / a
        a, b = int(a), int(b)
        if a < 1 or b < 1:
            return
        tmp = np.zeros((h, w), dtype=np.uint8)
        center = (np.random.randint(0, w), np.random.randint(0, h))
        angle = np.random.randint(0, 180)
        tmp = cv2.ellipse(tmp, center, (a, b), angle, 0, 360, 1, -1)
        mask[y:y+h, x:x+w] += tmp


    def __call__(self, image, target):
        if self.prob == 0:
            return image, target
        mode, image = image.mode, np.array(image)
        mask = np.zeros(image.shape[:-1], dtype=np.uint8)
        for ind in range(len(target)):

            if target.mode == 'xyxy':
                x, y, x2, y2 = target.bbox[ind]
                w, h = x2-x, y2-y
            else:
                assert target.mode == 'xywh'
                x,y,w,h = target.bbox[ind]

            x, y, w, h = int(x), int(y), int(w), int(h)

            if w*h < self.min_area_to_mask:
                continue

            if random.random() >= self.prob:
                continue

            try:
                if random.random() < 0.5:
                    self._ellipse_mask(mask, x, y, w, h)
                else:
                    self._line_mask(mask, x, y, w, h)
            except:
                continue
        mask = mask == 1
        if random.random() < 0.5:
            distorted = cv2.blur(image, (7, 7))  # mask by bluring
        else:
            distorted = np.ones_like(image) * \
                        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # mask by random colors
        image[mask] = distorted[mask]

        image = Image.fromarray(image, mode)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
