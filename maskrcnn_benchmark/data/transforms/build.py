# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        crop_prob = cfg.INPUT.CROP_PROB_TRAIN
        flip_prob = cfg.INPUT.FLIP_PROB_TRAIN
        transform_list = [
            T.Resize(min_size, max_size),
            T.RandomResizeCrop(crop_prob, (min_size, min_size)),
            T.RandomHorizontalFlip(flip_prob),
        ]
    else:
        transform_list = []

    if 'coco' in cfg.DATASETS.TRAIN or 'voc' in cfg.DATASETS.TRAIN:
        transform_list.append(
            T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD,
                to_bgr255=cfg.INPUT.TO_BGR255
            )
        )

    transform_list.append(T.ToTensor())
    transform = T.Compose(transform_list)
    return transform
