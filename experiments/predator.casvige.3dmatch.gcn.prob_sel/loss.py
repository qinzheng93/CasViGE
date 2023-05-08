from typing import Callable

import torch
import torch.nn as nn

from vision3d.loss import CircleLoss, WeightedBCELoss
from vision3d.ops import apply_transform, pairwise_distance, random_choice
from vision3d.ops.matching import extract_correspondences_from_feats
from vision3d.ops.metrics import evaluate_binary_classification


class DescriptorLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.circle_loss = CircleLoss(
            cfg.loss.circle_loss.positive_margin,
            cfg.loss.circle_loss.negative_margin,
            cfg.loss.circle_loss.positive_optimal,
            cfg.loss.circle_loss.negative_optimal,
            cfg.loss.circle_loss.log_scale,
        )

        self.max_correspondences = cfg.loss.circle_loss.max_correspondences
        self.pos_radius = cfg.loss.circle_loss.positive_radius
        self.neg_radius = cfg.loss.circle_loss.negative_radius
        self.weight = cfg.loss.circle_loss.weight
        self.eps = 1e-10

    @torch.no_grad()
    def get_recall(self, gt_corr_mat, fdist_mat):
        # Get feature match recall, divided by number of points which has inlier matches
        num_pos_points = torch.gt(gt_corr_mat.sum(-1), 0).float().sum() + 1e-12
        src_indices = torch.arange(fdist_mat.shape[0]).cuda()
        src_nn_indices = fdist_mat.min(-1)[1]
        pred_corr_mat = torch.zeros_like(fdist_mat)
        pred_corr_mat[src_indices, src_nn_indices] = 1.0
        recall = (pred_corr_mat * gt_corr_mat).sum() / num_pos_points
        return recall

    def forward(self, data_dict, output_dict):
        src_corr_indices = data_dict["src_corr_indices"]
        tgt_corr_indices = data_dict["tgt_corr_indices"]
        transform = data_dict["transform"]

        src_points = output_dict["src_points"]
        tgt_points = output_dict["tgt_points"]
        src_feats = output_dict["src_feats"]
        tgt_feats = output_dict["tgt_feats"]

        src_points = apply_transform(src_points, transform)

        if src_corr_indices.shape[0] > self.max_correspondences:
            sel_indices = random_choice(src_corr_indices.shape[0], size=self.max_correspondences, replace=False)
            src_corr_indices = src_corr_indices[sel_indices]
            tgt_corr_indices = tgt_corr_indices[sel_indices]
        src_corr_points = src_points[src_corr_indices]
        tgt_corr_points = tgt_points[tgt_corr_indices]
        src_corr_feats = src_feats[src_corr_indices]
        tgt_corr_feats = tgt_feats[tgt_corr_indices]

        fdist_mat = pairwise_distance(src_corr_feats, tgt_corr_feats, normalized=True, squared=False)
        dist_mat = pairwise_distance(src_corr_points, tgt_corr_points, squared=False, strict=True)
        pos_masks = torch.lt(dist_mat, self.pos_radius)
        neg_masks = torch.gt(dist_mat, self.neg_radius)

        loss = self.circle_loss(pos_masks, neg_masks, fdist_mat) * self.weight
        recall = self.get_recall(pos_masks.float(), fdist_mat)

        return loss, recall


class OverlapLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = cfg.loss.overlap_loss.weight
        self.weighted_bce_loss = WeightedBCELoss()

    def forward(self, data_dict, output_dict):
        src_corr_indices = data_dict["src_corr_indices"]
        tgt_corr_indices = data_dict["tgt_corr_indices"]
        src_scores = output_dict["src_overlap_scores"]
        tgt_scores = output_dict["tgt_overlap_scores"]

        src_pos_indices = torch.unique(src_corr_indices)
        tgt_pos_indices = torch.unique(tgt_corr_indices)

        src_labels = torch.zeros_like(src_scores)
        src_labels[src_pos_indices] = 1.0
        tgt_labels = torch.zeros_like(tgt_scores)
        tgt_labels[tgt_pos_indices] = 1.0

        scores = torch.cat([src_scores, tgt_scores], dim=0)
        labels = torch.cat([src_labels, tgt_labels], dim=0)

        loss = self.weighted_bce_loss(scores, labels) * self.weight

        precision, recall = evaluate_binary_classification(scores, labels)

        return loss, precision, recall


class SaliencyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pos_radius = cfg.loss.saliency_loss.positive_radius
        self.weight = cfg.loss.saliency_loss.weight
        self.weighted_bce_loss = WeightedBCELoss()

    def forward(self, data_dict, output_dict):
        src_corr_indices = data_dict["src_corr_indices"]
        tgt_corr_indices = data_dict["tgt_corr_indices"]
        transform = data_dict["transform"]

        src_points = output_dict["src_points"]
        tgt_points = output_dict["tgt_points"]
        src_scores = output_dict["src_saliency_scores"]
        tgt_scores = output_dict["tgt_saliency_scores"]
        src_feats = output_dict["src_feats"]
        tgt_feats = output_dict["tgt_feats"]

        src_points = apply_transform(src_points, transform)

        src_pos_indices = torch.unique(src_corr_indices)
        tgt_pos_indices = torch.unique(tgt_corr_indices)
        src_pos_points = src_points[src_pos_indices]
        tgt_pos_points = tgt_points[tgt_pos_indices]
        src_pos_scores = src_scores[src_pos_indices]
        tgt_pos_scores = tgt_scores[tgt_pos_indices]
        src_pos_feats = src_feats[src_pos_indices]
        tgt_pos_feats = tgt_feats[tgt_pos_indices]

        similarity_scores = torch.einsum("nc,mc->nm", src_pos_feats, tgt_pos_feats)
        src_nn_indices = similarity_scores.max(1)[1]
        src_nn_distances = torch.linalg.norm(src_pos_points - tgt_pos_points[src_nn_indices], dim=1)
        tgt_nn_indices = similarity_scores.max(0)[1]
        tgt_nn_distances = torch.linalg.norm(tgt_pos_points - src_pos_points[tgt_nn_indices], dim=1)
        src_pos_labels = torch.lt(src_nn_distances, self.pos_radius).float()
        tgt_pos_labels = torch.lt(tgt_nn_distances, self.pos_radius).float()

        labels = torch.cat([src_pos_labels, tgt_pos_labels], dim=0)
        scores = torch.cat([src_pos_scores, tgt_pos_scores], dim=0)

        loss = self.weighted_bce_loss(scores, labels) * self.weight

        precision, recall = evaluate_binary_classification(scores, labels)

        return loss, precision, recall


class LossFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.overlap_loss = OverlapLoss(cfg)
        self.saliency_loss = SaliencyLoss(cfg)
        self.descriptor_loss = DescriptorLoss(cfg)

    def forward(self, data_dict, output_dict):
        o_loss, o_precision, o_recall = self.overlap_loss(data_dict, output_dict)
        s_loss, s_precision, s_recall = self.saliency_loss(data_dict, output_dict)
        d_loss, d_recall = self.descriptor_loss(data_dict, output_dict)

        loss = o_loss + s_loss + d_loss

        return {
            "loss": loss,
            "o_loss": o_loss,
            "o_precision": o_precision,
            "o_recall": o_recall,
            "s_loss": s_loss,
            "s_precision": s_precision,
            "s_recall": s_recall,
            "d_loss": d_loss,
            "d_recall": d_recall,
        }


class EvalFunction(Callable):
    def __init__(self, cfg):
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.inlier_ratio_threshold = cfg.eval.inlier_ratio_threshold
        self.num_keypoints = cfg.eval.num_keypoints

    def __call__(self, data_dict, output_dict):
        transform = data_dict["transform"]
        src_points = output_dict["src_points"]
        tgt_points = output_dict["tgt_points"]
        src_scores = output_dict["src_scores"]
        tgt_scores = output_dict["tgt_scores"]
        src_feats = output_dict["src_feats"]
        tgt_feats = output_dict["tgt_feats"]

        src_points = apply_transform(src_points, transform)

        src_scores = src_scores / src_scores.sum()
        if src_points.shape[0] > self.num_keypoints:
            src_key_indices = random_choice(src_points.shape[0], self.num_keypoints, replace=False, p=src_scores)
            src_points = src_points[src_key_indices]
            src_feats = src_feats[src_key_indices]

        tgt_scores = tgt_scores / tgt_scores.sum()
        if tgt_points.shape[0] > self.num_keypoints:
            tgt_key_indices = random_choice(tgt_points.shape[0], self.num_keypoints, replace=False, p=tgt_scores)
            tgt_points = tgt_points[tgt_key_indices]
            tgt_feats = tgt_feats[tgt_key_indices]

        src_corr_indices, tgt_corr_indices = extract_correspondences_from_feats(
            src_feats, tgt_feats, mutual=True, normalized=True
        )
        src_corr_points = src_points[src_corr_indices]
        tgt_corr_points = tgt_points[tgt_corr_indices]
        corr_distances = torch.linalg.norm(src_corr_points - tgt_corr_points, dim=1)

        ir = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        fmr = torch.gt(ir, self.inlier_ratio_threshold).float()

        return {"IR": ir, "FMR": fmr}
