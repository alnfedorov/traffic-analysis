# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) Aleksandr Fedorov. Blur, brightness, contrast and instance masking augmentations.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness_prob = 0.75
        contrast_prob = 0.75
        instance_masking_prob = 0.75
        blur_prob = 0.4
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness_prob = 0
        instance_masking_prob = 0
        contrast_prob = 0
        blur_prob = 0


    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.BrightnessAdjust(brightness_prob),
            T.ContrastAdjust(contrast_prob),
            T.InstanceMasking(instance_masking_prob),
            T.MedianBlur(blur_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
