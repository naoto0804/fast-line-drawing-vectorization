import io
import random
import re
import xml.etree.ElementTree as et
from pathlib import Path

import cairosvg
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps

from maskrcnn_benchmark.data.transforms.build import build_transforms
from maskrcnn_benchmark.data.transforms.transforms import Compose
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

ROOT = "../../data"


def get_dataset(cfg, dataset="ch", subset="train", is_train=True):
    assert isinstance(dataset, str)
    assert isinstance(subset, str)
    assert isinstance(is_train, bool)

    # TODO: support multi-class
    # we train a single network on a set of classes (BASEBALL, CAT, CHANDELIER and ELEPHANT), which we call the MULTI-CLASS dataset. We then compute the testing error on a set of unseen classes (BACKPACK and BICYCLE).
    datasets_dict = {
        "ch": ChineseDataset,
        "kanji": KanjiDataset,
        "line": LineDataset,
        "cat": QDrawDataset,
        "baseball": QDrawDataset,
        "multi": QDrawDataset,
    }
    assert subset in ["train", "test"]
    assert dataset in datasets_dict.keys()
    root = str(Path(ROOT) / dataset)
    transforms = build_transforms(cfg, is_train)
    return datasets_dict[dataset](root, subset, transforms)


class BaseDataset(torch.utils.data.Dataset):
    CLASSES = ("BG", "FG")

    def __init__(self, data_dir, split, transforms=None, size=64,
                 canvas_size=1024):
        assert isinstance(data_dir, str)
        assert isinstance(split, str)
        assert isinstance(transforms, Compose)
        assert isinstance(size, int)
        assert isinstance(canvas_size, int)

        self.data_dir = Path(data_dir)
        self.split = split
        # self.png_name = self.data_dir / "png" / "{:s}.png"
        # self.svg_name = self.data_dir / "svg" / "{:s}.svg"

        with (self.data_dir / "{:s}.txt".format(split)).open("r") as f:
            self.ids = [_.strip() for _ in f.readlines()]

        self.size = size
        self.canvas_size = canvas_size
        self.transforms = transforms

    def get_groundtruth(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        image, boxlist = self.get_groundtruth(idx)
        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)
        boxlist = boxlist.clip_to_image(remove_empty=True)
        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        return {"height": self.size, "width": self.size}

    @staticmethod
    def transform_y(y):
        return y

    @staticmethod
    def get_bbox(mask, mode="xyxy"):
        assert isinstance(mask, np.ndarray)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if mode == "xyxy":
            return [cmin, rmin, cmax, rmax]
        else:
            raise NotImplementedError

    def svg_to_png(self, svg):
        image = cairosvg.svg2png(bytestring=svg)
        image = Image.open(io.BytesIO(image)).split()[-1].convert("RGB")
        image = ImageOps.invert(image)
        assert (image.size == (self.size, self.size))
        return image


class ChineseDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=64,
                 canvas_size=1024):
        super().__init__(data_dir, split, transforms, size, canvas_size)

    def get_groundtruth(self, idx):
        bboxes = []
        masks = []

        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        global shift_x
        global shift_y
        global max_size
        shift_max = int(self.canvas_size * 0.1)
        shift_x = random.randint(-shift_max, shift_max)
        shift_y = random.randint(-shift_max, shift_max)
        max_size = self.canvas_size

        def _replace_num(match):
            x, y = [int(t) for t in match.group().split(" ")]
            x = max(0, min(max_size - 1, x + shift_x))
            y = max(0, min(max_size - 1, y + shift_y))
            # y = max(900 - max_size, min(900, y + shift_y))
            return "{:d} {:d}".format(x, y)

        def _replace_path(match):
            return re.sub(r'-?[0-9]+ [0-9]+', _replace_num, match.group())

        # svg = re.sub(r'<path d=".*"></path>', _replace_path, svg)
        # print(self.ids[idx], svg, "\n")

        num_paths = len(et.fromstring(svg)[0])
        for i in range(num_paths):
            svg_xml = et.fromstring(svg)
            svg_xml[0][0] = svg_xml[0][i]
            del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)

            try:
                bboxes.append(self.get_bbox(mask, "xyxy"))
            except IndexError:
                continue
            masks.append(mask.astype(np.uint8))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist


class KanjiDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=64,
                 canvas_size=109):
        super().__init__(data_dir, split, transforms, size, canvas_size)

    def get_groundtruth(self, idx):
        bboxes = []
        masks = []

        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        pid = 0
        num_paths = 0
        while pid != -1:
            pid = svg.find('path id', pid + 1)
            num_paths = num_paths + 1
        num_paths = num_paths - 1  # uncount last one

        for i in range(num_paths):
            svg_one = svg
            pid = len(svg_one)

            for j in range(num_paths):
                pid = svg_one.rfind('path id', 0, pid)
                if j != i:
                    id_start = svg_one.rfind('>', 0, pid) + 1
                    id_end = svg_one.find('/>', id_start) + 2
                    svg_one = svg_one[:id_start] + svg_one[id_end:]

            y_png = cairosvg.svg2png(bytestring=svg_one.encode('utf-8'))
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            bboxes.append(self.get_bbox(mask, "xyxy"))
            masks.append(mask.astype(np.uint8))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist


class LineDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=64,
                 canvas_size=64):
        super().__init__(data_dir, split, transforms, size, canvas_size)

    def get_groundtruth(self, idx: int):
        bboxes = []
        masks = []

        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        svg_xml = et.fromstring(svg)

        for i in range(len(svg_xml[0])):
            svg_xml = et.fromstring(svg)
            svg_xml[0][0] = svg_xml[0][i]
            del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            assert mask.shape == (self.size, self.size), (self.ids[idx])
            bboxes.append(self.get_bbox(mask, "xyxy"))
            masks.append(mask.astype(np.uint8))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist


class QDrawDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=128,
                 canvas_size=128):
        super().__init__(data_dir, split, transforms, size, canvas_size)
        ng_ids = ["4585937717690368", "4583140179836928",  # cat
                  "6231511248404480", "6442058833199104",  # cat
                  "5327861537832960"]  # baseball
        for ng_id in ng_ids:
            if ng_id in self.ids:
                self.ids.remove(ng_id)

    def get_groundtruth(self, idx):
        bboxes = []
        masks = []

        svg_name = self.data_dir / self.split / "{:s}.svg".format(
            self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        num_paths = svg.count('polyline')

        for i in range(1, num_paths + 1):
            svg_xml = et.fromstring(svg)
            svg_xml[1] = svg_xml[i]
            del svg_xml[2:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            try:
                bboxes.append(self.get_bbox(mask, "xyxy"))
            except:
                continue
            masks.append(mask.astype(np.uint8))

        # if there is completely no boxes, add dummy boxes and masks
        if len(bboxes) == 0:
            print(self.ids[idx], et.tostring(svg_xml, method='xml'))
            dummy_mask = np.zeros(mask.shape, dtype=np.uint8)
            dummy_mask[0:4, 0:4] = 1
            masks.append(dummy_mask.astype(np.uint8))
            bboxes.append(self.get_bbox(dummy_mask, "xyxy"))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist
