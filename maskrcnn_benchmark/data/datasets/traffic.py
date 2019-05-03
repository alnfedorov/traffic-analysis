import torch
import os
import gc
import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO, maskUtils


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if _has_only_empty_bbox(anno):
        return False
    return False


class TrafficDataset(Dataset):
    def __init__(self, ann_file, root, to_contiguous_class_mapping, to_json_class_mapping, transforms):
        self.coco = COCO(ann_file)
        self.class_mapping = to_contiguous_class_mapping
        MASK = Image.open(os.path.join(root, 'mask.png'))

        # Mask used to keep only visible roadbed
        tmp = np.array(MASK)
        data_points = np.argwhere(tmp)
        self.min_y, self.min_x = data_points.min(axis=0)
        self.max_y, self.max_x = data_points.max(axis=0) + 1
        tmp = tmp[self.min_y:self.max_y, self.min_x:self.max_x]
        assert tmp.shape == (769, 1920), tmp.shape
        MASK = Image.fromarray(tmp, MASK.mode)

        # filter images without detection annotations
        self.ids, self.images = [], []
        for img_id in sorted(self.coco.imgs.keys()):
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                path = os.path.join(root, self.coco.imgs[img_id]['file_name'])

                try:
                    # open and crop
                    image = Image.open(path)
                    image = Image.fromarray(np.array(image)[self.min_y:self.max_y, self.min_x:self.max_x], image.mode)

                    zeros = Image.fromarray(np.zeros_like(image), image.mode)
                    image = Image.composite(image, zeros, mask=MASK)
                except Exception as e:
                    print("Failed to load image ", path, e)
                    continue

                self.images.append(image)
                self.ids.append(img_id)

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)} # inner id to json id

        self.contiguous_category_id_to_json_id = copy.deepcopy(to_json_class_mapping) # inner class -> json class name
        for class_id, class_name in self.contiguous_category_id_to_json_id.items():
            self.contiguous_category_id_to_json_id[class_id] = \
                [x for x in self.coco.cats.values() if x['name'] == class_name][0]['id'] # inner contiguous class to json class

        self.transforms = transforms

        # Update bboxes to avoid coco-annotator bugs with conversion bb coordinates
        from tqdm import tqdm
        for k, v in tqdm(list(self.coco.anns.items())):
            rle = self.coco.annToRLE(v)
            before = v['bbox']
            v['bbox'] = maskUtils.toBbox(rle)
            if sum(abs(before - v['bbox'])) > 1:
                print(f"Changed {before}->{v['bbox']}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[idx]

        coco_idx = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds([coco_idx]))

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.class_mapping[self.coco.cats[c]['name']] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.crop([self.min_x, self.min_y, self.max_x, self.max_y]).clip_to_image(remove_empty=True)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.ids[index]
        img_data = self.coco.imgs[img_id]
        img_data["crop_x"], img_data["crop_y"] = self.min_x, self.min_y
        img_data["crop_w"], img_data["crop_h"] = self.max_x - self.min_x, self.max_y - self.min_y
        return img_data
