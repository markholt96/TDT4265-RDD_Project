from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *


def build_transforms(cfg, is_train=True, augment=False):
    if augment:
        transform = [
            ConvertFromInts(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
        ]
    if is_train:
        transform = [
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
