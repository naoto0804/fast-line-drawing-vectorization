# Written by Naoto Inoue, 2019-04
import argparse
import time

import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.datasets import get_dataset
from scripts.demo import Tester, evaluate_iou

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--conf_thresh", default=0.5, type=float)
    parser.add_argument("--mask_thresh", default=0.5, type=float)
    parser.add_argument("--no_force_label", action='store_true')
    parser.add_argument("--N", default=0, type=int, help="Number of samples used for evaluation.")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    name = cfg.DATASETS.TEST[0].split("_")[0]
    dataset = get_dataset(cfg, name, "test", False)
    demo = Tester(cfg, min_image_size=dataset.get_img_info(0)["height"],
                  conf_thresh=args.conf_thresh, mask_thresh=args.mask_thresh,
                  ckpt=args.ckpt)

    iou_list, pred_gt_list, time_list = [], [], []
    N = len(dataset) if args.N <= 0 else args.N

    for i in range(N):
        start_time = time.time()
        image, boxlist, idx = dataset[i]
        valid_pred_bins = demo.predict(image, not args.no_force_label)
        time_list.append(time.time() - start_time)

        gt_bins = [np.array(x.mask, dtype=np.bool) for x in
                   boxlist.get_field("mask").masks]
        iou_list_per_image = evaluate_iou(gt_bins, valid_pred_bins)

        avg_iou = sum(iou_list_per_image) / len(iou_list_per_image)
        num_pred = sum(p.sum() for p in valid_pred_bins)
        gt_pred = sum(g.sum() for g in gt_bins)
        text = "ID {:s} AVG {:.3f} PRED/GT {:.2f} PRED {:d} GT {:d}"
        # print(text.format(dataset.ids[i], avg_iou, num_pred / gt_pred,
        #                   len(valid_pred_bins), len(gt_bins)))
        iou_list.append(avg_iou)
        pred_gt_list.append(num_pred / gt_pred)
        if i % 100 == 0:
            print(i)
    print("Time per image {:.3f}".format(sum(time_list) / len(time_list)))
    print("Total IoU: {:.3f}".format(sum(iou_list) / len(iou_list)))
    print(
        "Total PRED/GT: {:.3f}".format(sum(pred_gt_list) / len(pred_gt_list)))
