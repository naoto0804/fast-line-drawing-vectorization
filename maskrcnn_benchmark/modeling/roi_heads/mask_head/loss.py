# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, \
    Mask


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used

        # there sometimes is shape mismatch
        # print(cropped_mask.mask.shape, cropped_mask.size, proposal)
        # if the edge of the proposal is too short, enlarge it
        # make sure 0 <= x <= W - 1, 0 <= y <= H - 1
        W, H = segmentation_mask.size
        if proposal[2] - proposal[0] <= 2.0:
            proposal[0] = max(proposal[0] - 1.0, 0.0)
            proposal[2] = min(proposal[2] + 1.0, W - 1)
        if proposal[3] - proposal[1] <= 2.0:
            proposal[1] = max(proposal[1] - 1.0, 0.0)
            proposal[3] = min(proposal[3] + 1.0, H - 1)
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size,
                 use_overlap=False):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
            use_overlap (bool)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.use_overlap = use_overlap

        self.mae_loss = nn.L1Loss()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "mask"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        overlaps = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("mask")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals,
                self.discretization_size
            )
            labels.append(labels_per_image)
            masks.append(masks_per_image)

            if self.use_overlap:
                size = targets_per_image.size
                boxlist_mask = targets_per_image.get_field('mask')
                overlap = torch.zeros(boxlist_mask.masks[0].size)
                for m in boxlist_mask.masks:
                    overlap += m.mask

                # overlap or not classification
                mask_overlap = Mask((overlap >= 2.0).float(), size, "mask")
                # duplicate according to the number of positive proposals
                overlap_per_image = project_masks_on_boxes(
                    SegmentationMask(
                        [mask_overlap] * len(positive_proposals), size,
                        "mask"),
                    positive_proposals, self.discretization_size
                )
                overlaps.append(overlap_per_image)

        if self.use_overlap:
            return labels, masks, overlaps
        else:
            return labels, masks

    def __call__(self, proposals, mask_logits, targets, overlap_logits=None):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])
            overlap_logits (Tensor, optional)

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        if self.use_overlap:
            labels, mask_targets, overlap_targets = \
                self.prepare_targets(proposals, targets)
        else:
            labels, mask_targets = self.prepare_targets(proposals, targets)

        labels = cat(labels, dim=0)
        mask_targets = cat(mask_targets, dim=0)

        positive_inds = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[positive_inds]

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[positive_inds, labels_pos], mask_targets
        )
        assert len(mask_logits) == len(positive_inds) == len(labels)
        if self.use_overlap:
            overlap_loss = 0.0
            n_proposals = [len(p) for p in proposals]
            start_inds = [0] + list(accumulate(n_proposals))[:-1]
            for i, (start, step) in enumerate(zip(start_inds, n_proposals)):
                overlap_loss += F.binary_cross_entropy_with_logits(
                    overlap_logits[start:start + step, 0], overlap_targets[i]
                )
            overlap_loss /= len(overlap_targets)
            mask_loss += overlap_loss * 0.1

        ## Added
        # accumulation of masks for each image
        # device = proposals[0].bbox.device
        # n_proposals = [len(p) for p in proposals]
        # start_inds = [0] + list(accumulate(n_proposals))[:-1]
        # accm_targets = []
        # H, W = mask_logits.shape[2:]
        #
        # for t in targets:
        #     masks = t.get_field('mask').masks
        #     accm = torch.zeros((H, W))
        #     for m in masks:
        #         accm += m.resize((W, H)).mask
        #     accm_targets.append(accm)
        #
        # assert len(positive_inds) == sum(n_proposals)
        # assert labels_pos.sum().item() == len(labels_pos)  # labels are all 1
        # splitted_mask_probs = []
        # for (start, step) in zip(start_inds, n_proposals):
        #     probs = mask_logits.narrow(0, start, step).sigmoid()
        #     splitted_mask_probs.append(probs.sum(dim=0)[1])
        #
        # splitted_mask_probs = torch.stack(splitted_mask_probs, dim=0)
        # accm_targets = torch.stack(accm_targets, dim=0).to(device)
        # loss = self.mae_loss(splitted_mask_probs, accm_targets)

        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    use_overlap = True if cfg.MODEL.ROI_MASK_HEAD.OVERLAP else False
    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION, use_overlap
    )

    return loss_evaluator
