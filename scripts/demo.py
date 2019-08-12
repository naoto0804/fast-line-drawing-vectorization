# Written by Naoto Inoue, 2019-04
import argparse
import itertools
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms as T

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.datasets import get_dataset
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def iou(array_1, array_2):
    assert array_1.shape == array_2.shape
    assert array_1.dtype == np.bool
    assert array_2.dtype == np.bool
    intersect = (array_1 & array_2).sum()
    union = (array_1 | array_2).sum()
    return intersect / union


def evaluate_iou(gt_bins, valid_pred_bins):
    iou_list_per_image = []
    for valid_pred_bin in valid_pred_bins:
        iou_list_per_pred = []
        for gt_bin in gt_bins:
            iou_list_per_pred.append(iou(gt_bin, valid_pred_bin))
        if len(iou_list_per_pred) > 0:
            iou_list_per_image.append(max(iou_list_per_pred))
        else:
            iou_list_per_image.append(0.0)
    return iou_list_per_image


def test_evaluate_iou():
    gt_bin_1 = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ]).astype(np.bool)
    valid_pred_bin_1 = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ]).astype(np.bool)
    valid_pred_bin_2 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]).astype(np.bool)

    print(evaluate_iou([gt_bin_1], [valid_pred_bin_1]))  # 1 / 7
    print(
        evaluate_iou([gt_bin_1], [valid_pred_bin_1, valid_pred_bin_2]))  # 0.5


def render_svg(bins):
    cmap = plt.get_cmap('jet')
    cnorm = colors.Normalize(vmin=0, vmax=len(bins) - 1)
    cscalarmap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)

    first_svg = True
    _, tgt_svg_path = tempfile.mkstemp(suffix='.svg')
    tgt_svg_path = Path(tgt_svg_path)

    for i, valid_pred_bin in enumerate(bins):
        _, tmp_bmp_path = tempfile.mkstemp(suffix='.bmp')
        tmp_bmp_path = Path(tmp_bmp_path)
        cond = (valid_pred_bin == True) & (
                image.numpy().mean(axis=0) < 0.5)
        tmp_bmp_img = np.where(cond, 255, 0)
        imageio.imsave(str(tmp_bmp_path), tmp_bmp_img.astype(np.uint8))
        color = np.asarray(cscalarmap.to_rgba(i))
        color *= 255
        color_hex = "#{:02x}{:02x}{:02x}".format(*[int(c) for c in color])
        exe = "/home/inoue/build/potrace-1.15.linux-x86_64/potrace"
        subprocess.call([exe, '-s', '-i', '-C' + color_hex, tmp_bmp_path])
        tmp_bmp_path = Path(tmp_bmp_path)
        tmp_svg_path = tmp_bmp_path.with_suffix(".svg")
        if first_svg:
            shutil.move(str(tmp_svg_path), str(tgt_svg_path))
            first_svg = False
        else:
            with tgt_svg_path.open("r") as f_tgt:
                tgt_svg = f_tgt.read()
            with tmp_svg_path.open("r") as f_src:
                src_svg = f_src.read()

            path_start = src_svg.find('<g')
            path_end = src_svg.find('</svg>')

            insert_pos = tgt_svg.find('</svg>')
            tgt_svg = tgt_svg[:insert_pos] + \
                      src_svg[path_start:path_end] + tgt_svg[insert_pos:]
            with tgt_svg_path.open("w") as f_tgt:
                f_tgt.write(tgt_svg)
            tmp_svg_path.unlink()
        tmp_bmp_path.unlink()

    # set opacity 0.5 to see overlaps
    with tgt_svg_path.open("r") as f_tgt:
        tgt_svg = f_tgt.read()
    insert_pos = tgt_svg.find('<g')
    tgt_svg = tgt_svg[:insert_pos] + \
              '<g fill-opacity="0.5">' + tgt_svg[insert_pos:]
    insert_pos = tgt_svg.find('</svg>')
    tgt_svg = tgt_svg[:insert_pos] + '</g>' + tgt_svg[insert_pos:]
    with tgt_svg_path.open("w") as f_tgt:
        f_tgt.write(tgt_svg)
    return tgt_svg_path


class Tester(object):
    CLASSES = ("BG", "FG")

    def __init__(self, cfg, conf_thresh=0.7, mask_thresh=0.5, masks_per_dim=2,
                 min_image_size=224, ckpt=None):
        assert 0.0 <= conf_thresh <= 1.0 and isinstance(conf_thresh, float)
        assert 0.0 <= mask_thresh <= 1.0 and isinstance(mask_thresh, float)
        assert masks_per_dim > 0 and isinstance(masks_per_dim, int)
        assert min_image_size > 0 and isinstance(min_image_size, int)

        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model,
                                             save_dir=save_dir)
        if ckpt:
            checkpointer.load_model(checkpointer.load_file(ckpt))
        else:
            _ = checkpointer.load()

        self.transforms = self.build_transform()
        self.masker = Masker(threshold=mask_thresh, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.conf_thresh = conf_thresh
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        transform = T.Compose([T.ToTensor()])
        return transform

    def visualize_png(self, image, force_label=True, overlay_box=False,
                      overlay_mask=True):
        assert isinstance(image, torch.Tensor)
        assert isinstance(force_label, bool)
        assert isinstance(overlay_box, bool)
        assert isinstance(overlay_mask, bool)

        result = (image.numpy() * 255).astype(np.uint8)
        result = result.transpose(1, 2, 0)[..., ::-1]

        valid_pred_bins = self.predict(image, force_label)
        if overlay_box:
            result = self.overlay_boxes(result, valid_pred_bins)
        if overlay_mask:
            result = self.overlay_mask(result, valid_pred_bins)
        return result

    def visualize_svg(self, image, boxlist, force_label=True):
        assert isinstance(image, torch.Tensor)
        assert isinstance(boxlist, BoxList)
        assert isinstance(force_label, bool)

        gt_bins = [np.array(x.mask, dtype=np.bool) for x in
                   boxlist.get_field("mask").masks]
        valid_pred_bins = self.predict(image, force_label)
        iou_list = evaluate_iou(gt_bins, valid_pred_bins)
        iou = sum(iou_list) / len(iou_list)
        return render_svg(valid_pred_bins), render_svg(gt_bins), iou

    def predict(self, image, force_label=True):
        assert isinstance(image, torch.Tensor)
        assert isinstance(force_label, bool)

        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        # get a bibary mask of non-white pixels
        valid_bin = np.all((image.numpy() != 1.0), axis=0).astype(np.bool)
        pred_bins = [np.array(x.squeeze(0), dtype=np.bool) for x in
                     top_predictions.get_field("mask")]

        # filter out predictions on non-white pixels
        valid_pred_bins = []
        for bin in pred_bins:
            if force_label:
                valid_pred_bins.append(valid_bin & bin)
            else:
                valid_pred_bins.append(bin)

        if len(valid_pred_bins) == 0:
            return valid_bin[np.newaxis, ...]
        else:
            valid_pred_bins = np.stack(valid_pred_bins)

        # if IoU is larger than a fixed value, remove smaller segments
        comb = list(itertools.combinations(enumerate(valid_pred_bins), 2))
        removed_inds = []
        for (i, bin_i), (j, bin_j) in comb:
            if i in removed_inds or j in removed_inds:
                continue
            # if iou(bin_i, bin_j) > 0.3:
            if iou(bin_i, bin_j) > 0.2:
                # The bigger, the better
                if bin_i.mean() > bin_j.mean():
                    removed_inds.append(j)
                else:
                    removed_inds.append(i)
        if len(removed_inds) > 0:
            inds = np.delete(np.arange(len(valid_pred_bins)), removed_inds)
            valid_pred_bins = valid_pred_bins[inds]

        # TODO: nearest neighbour trying to prefer connected labels
        if force_label:
            # search for pixels that are not labeled at all
            acc_pred_bin = np.any(valid_pred_bins, axis=0)

            # filter out predictions on non-white pixels
            valid_pred_coords = \
                np.array([(y, x) for y, x in zip(*np.where(
                    (valid_bin == True) & (acc_pred_bin == True)))])
            valid_non_pred_coords = \
                np.array([(y, x) for y, x in zip(*np.where(
                    (valid_bin == True) & (acc_pred_bin == False)))])

            if len(valid_non_pred_coords) > 0:
                neigh = NearestNeighbors()
                try:
                    neigh.fit(valid_pred_coords)
                except ValueError:
                    # ValueError: Expected 2D array, got 1D array instead:
                    print("L97 ValueError")
                    return valid_pred_bins
                ind = np.array([int(_.squeeze()) for _ in
                                neigh.kneighbors(valid_non_pred_coords, 1,
                                                 False)])
                src_coords = valid_pred_coords[ind]
                assert len(src_coords) == len(valid_non_pred_coords)
                src_y_coords, src_x_coords = \
                    [_.squeeze() for _ in np.split(src_coords, 2, axis=1)]
                tgt_y_coords, tgt_x_coords = \
                    [_.squeeze() for _ in
                     np.split(valid_non_pred_coords, 2, axis=1)]

                for valid_pred_bin in valid_pred_bins:
                    ind = np.where(
                        valid_pred_bin[src_y_coords, src_x_coords] == True)
                    if len(ind[0]) > 0 and tgt_x_coords.ndim > 0:
                        valid_pred_bin[
                            tgt_y_coords[ind], tgt_x_coords[ind]] = True
        return valid_pred_bins

    def compute_prediction(self, image):
        assert isinstance(image, torch.Tensor)
        # original_image = np.array(original_image)[..., ::-1]  # to opencv image
        image = image.unsqueeze(0)

        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image,
                                   self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = image.shape[-2:]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        assert isinstance(predictions, BoxList)

        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.conf_thresh).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def overlay_boxes(self, image, pred_bins):
        assert isinstance(image, np.ndarray)
        assert isinstance(pred_bins, np.ndarray)

        colors = [tuple(y * 255 for y in cm.jet(x)[:-1]) for x in
                  np.linspace(0.0, 1.0, len(pred_bins))]
        for pred_bin, color in zip(pred_bins, colors):
            if pred_bin.mean() == 0.0:
                continue
            rows = np.any(pred_bin, axis=1)
            cols = np.any(pred_bin, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            image = cv2.rectangle(
                image, (cmin, rmin), (cmax, rmax), tuple(color), 1)
        return image

    def overlay_mask(self, image, pred_bins):
        assert isinstance(image, np.ndarray)
        assert isinstance(pred_bins, np.ndarray)

        colors = [tuple(y * 255 for y in cm.jet(x)[:-1]) for x in
                  np.linspace(0.0, 1.0, len(pred_bins))]

        for pred_bin, color in zip(pred_bins, colors):
            contours, hierarchy = cv2.findContours(
                (pred_bin * 255).astype(np.uint8),
                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color, 1)
        return image


def load(file_name):
    pil_image = Image.open(file_name).convert('RGB')
    image = np.array(pil_image)[..., ::-1]
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--output_dir", type=str, default="tmp")
    parser.add_argument("--mode", default="png", choices=["png", "svg"], 
        help="Choose the type of the result")
    parser.add_argument("--conf_thresh", default=0.5, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument("--N", default=10, type=int, help="Number of result images.")
    parser.add_argument("--no_force_label", action="store_true", 
        help="Choose whether to do pre-processing consisting of two steps")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    name = cfg.DATASETS.TEST[0].split('_')[0]
    dataset = get_dataset(cfg, name, "test", False)
    demo = Tester(cfg, min_image_size=dataset.get_img_info(0)["height"],
                  conf_thresh=args.conf_thresh, mask_thresh=args.mask_thresh,
                  ckpt=args.ckpt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(len(dataset)):
        image, boxlist, idx = dataset[i]
        if args.mode == "png":
            pred = demo.visualize_png(image, not args.no_force_label)
            cv2.imwrite(
                str(output_dir / "{:s}.png".format(dataset.ids[i])), pred)
        else:
            old_svg, old_gt_svg, acc = \
                demo.visualize_svg(image, boxlist, not args.no_force_label)
            new_svg = output_dir / "{:s}.{:.2f}.svg".format(dataset.ids[i],
                                                            acc)
            new_gt_svg = output_dir / "{:s}_gt.svg".format(dataset.ids[i])
            shutil.move(str(old_svg), str(new_svg))
            shutil.move(str(old_gt_svg), str(new_gt_svg))

        if i == args.N:
            break
