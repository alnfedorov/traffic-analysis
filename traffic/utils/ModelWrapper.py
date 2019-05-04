# Partially based on the code from opencv demo in maskrcnn-benchmark

import torch
from torchvision import transforms as T
from maskrcnn_benchmark.data.transforms import transforms as TT

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.config import cfg


class ModelWrapper(object):
    def __init__(self, config_file, confidence_threshold=0.3, mask_threshold=0.5, mask_on=True):
        cfg.merge_from_file(config_file)
        cfg.MODEL.MASK_ON = mask_on
        cfg.freeze()

        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg).eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=cfg.OUTPUT_DIR)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()

        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

    def build_transform(self):
        cfg = self.cfg
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST

        # work around to reuse code in Transforms for images only
        class Normalize(TT.Normalize):
            def __call__(self, img):
                return super().__call__(img, None)[0]

        class Resize(TT.Resize):
            def __call__(self, img):
                return super().__call__(img, None)[0]


        normalize_transform = Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
        )
        resize = Resize(min_size, max_size)

        transform = T.Compose(
            [
                T.ToPILImage(),
                resize,
                T.ToTensor(),
                normalize_transform,
            ]
        )
        return transform

    def predict(self, images, project_mask=False):
        # apply pre-processing to image
        assert all(x.shape == images[0].shape for x in images)
        transformed = [self.transforms(x) for x in images]

        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(transformed, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        with torch.no_grad():
            predictions = self.model(image_list)

        predictions = [o.to(self.cpu_device) for o in predictions]
        prediction = [self.select_top_predictions(x) for x in predictions]

        height, width = images[0].shape[:-1]
        prediction = [x.resize((width, height)) for x in prediction]

        if project_mask:
            assert len(prediction) == 1, "not supported"
            prediction = self.project_masks(prediction[0])

        return prediction

    def project_masks(self, bboxlist):
        if bboxlist.has_field("mask"):
            masks = bboxlist.get_field("mask")
            masks = self.masker([masks], [bboxlist])[0]
            bboxlist.add_field("mask", masks)
        return bboxlist

    def select_top_predictions(self, predictions):
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]