import glob
import json
import os.path as osp
import time

import numpy as np

from vision3d.array_ops import (
    evaluate_correspondences,
    extract_correspondences_from_feats,
    isotropic_registration_error,
    weighted_procrustes,
)
from vision3d.datasets.registration.threedmatch.threedmatch_utils import (
    compute_transform_error,
    get_gt_logs_and_infos,
    get_num_fragments,
    get_scene_abbr,
    write_log_file,
)
from vision3d.utils.logger import get_logger
from vision3d.utils.open3d import registration_with_ransac_from_correspondences
from vision3d.utils.parser import get_default_parser
from vision3d.utils.summary_board import SummaryBoard

# isort: split
from config import make_cfg


def make_parser():
    parser = get_default_parser()
    parser.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch"], required=True, help="test benchmark")
    parser.add_argument("--method", choices=["lgr", "ransac", "svd"], required=True, help="registration method")
    parser.add_argument("--num_keypoints", type=int, required=True, help="number of keypoints")
    parser.add_argument("--num_correspondences", type=int, default=None, help="number of correspondences")
    parser.add_argument("--mutual", type=bool, default=True, help="mutual matching")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    return parser


def eval_one_epoch(args, cfg, logger):
    features_dir = osp.join(cfg.exp.cache_dir, args.benchmark)
    benchmark = args.benchmark

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter("recall")
    fine_matching_meter.register_meter("inlier_ratio")
    fine_matching_meter.register_meter("overlap")
    fine_matching_meter.register_meter("scene_recall")
    fine_matching_meter.register_meter("scene_inlier_ratio")
    fine_matching_meter.register_meter("scene_overlap")

    registration_meter = SummaryBoard()
    registration_meter.register_meter("recall")
    registration_meter.register_meter("mean_rre")
    registration_meter.register_meter("mean_rte")
    registration_meter.register_meter("median_rre")
    registration_meter.register_meter("median_rte")
    registration_meter.register_meter("scene_recall")
    registration_meter.register_meter("scene_rre")
    registration_meter.register_meter("scene_rte")

    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    scene_dirs = sorted(glob.glob(osp.join(features_dir, "*")))
    for scene_dir in scene_dirs:
        fine_matching_meter.reset_meter("scene_recall")
        fine_matching_meter.reset_meter("scene_inlier_ratio")
        fine_matching_meter.reset_meter("scene_overlap")

        registration_meter.reset_meter("scene_recall")
        registration_meter.reset_meter("scene_rre")
        registration_meter.reset_meter("scene_rte")

        scene_name = osp.basename(scene_dir)
        scene_abbr = get_scene_abbr(scene_name)
        num_fragments = get_num_fragments(scene_name)
        gt_dir = osp.join(cfg.data.dataset_dir, "metadata", "benchmarks", benchmark, scene_name)
        gt_indices, gt_logs, gt_infos = get_gt_logs_and_infos(gt_dir, num_fragments)

        estimated_transforms = []

        file_names = sorted(
            glob.glob(osp.join(scene_dir, "*.npz")),
            key=lambda x: [int(i) for i in osp.basename(x).split(".")[0].split("_")],
        )
        for file_name in file_names:
            tgt_frame, src_frame = [int(x) for x in osp.basename(file_name).split(".")[0].split("_")]

            data_dict = np.load(file_name)

            src_points = data_dict["src_points"]
            tgt_points = data_dict["tgt_points"]
            src_feats = data_dict["src_feats"]
            tgt_feats = data_dict["tgt_feats"]
            src_scores = data_dict["src_scores"]
            tgt_scores = data_dict["tgt_scores"]
            transform = data_dict["transform"]
            pcd_overlap = data_dict["overlap"]

            if src_scores.shape[0] > args.num_keypoints:
                src_scores = src_scores / src_scores.sum()
                src_key_indices = np.random.choice(src_scores.shape[0], args.num_keypoints, replace=False, p=src_scores)
                src_key_points = src_points[src_key_indices]
                src_key_feats = src_feats[src_key_indices]
            else:
                src_key_points = src_points
                src_key_feats = src_feats

            if tgt_scores.shape[0] > args.num_keypoints:
                tgt_scores = tgt_scores / tgt_scores.sum()
                tgt_key_indices = np.random.choice(tgt_scores.shape[0], args.num_keypoints, replace=False, p=tgt_scores)
                tgt_key_points = tgt_points[tgt_key_indices]
                tgt_key_feats = tgt_feats[tgt_key_indices]
            else:
                tgt_key_points = tgt_points
                tgt_key_feats = tgt_feats

            src_corr_points, tgt_corr_points, corr_scores = extract_correspondences_from_feats(
                src_key_points, tgt_key_points, src_key_feats, tgt_key_feats, mutual=True, return_feat_dist=True
            )

            if args.num_correspondences is not None and corr_scores.shape[0] > args.num_correspondences:
                sel_indices = np.argsort(corr_scores)[: args.num_correspondences]
                src_corr_points = src_corr_points[sel_indices]
                tgt_corr_points = tgt_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]

            corr_scores = np.exp(-corr_scores)

            message = f"{scene_abbr}, src: {src_frame}, tgt: {tgt_frame}, OV: {pcd_overlap:.3f}"

            # 1. evaluate correspondences
            # 1.1 evaluate fine correspondences
            fine_matching_result_dict = evaluate_correspondences(
                src_corr_points, tgt_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
            )

            inlier_ratio = fine_matching_result_dict["inlier_ratio"]
            overlap = fine_matching_result_dict["overlap"]

            fine_matching_meter.update("scene_inlier_ratio", inlier_ratio)
            fine_matching_meter.update("scene_overlap", overlap)
            fine_matching_meter.update("scene_recall", float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

            message += ", IR: {:.3f}".format(inlier_ratio)
            message += ", OV: {:.3f}".format(overlap)
            message += ", RS: {:.3f}".format(fine_matching_result_dict["distance"])
            message += ", NU: {}".format(corr_scores.shape[0])

            # 2. evaluate registration
            if args.method == "lgr":
                estimated_transform = data_dict["estimated_transform"]
            elif args.method == "ransac":
                estimated_transform = registration_with_ransac_from_correspondences(
                    src_corr_points,
                    tgt_corr_points,
                    distance_threshold=cfg.ransac.distance_threshold,
                    ransac_n=cfg.ransac.num_points,
                    num_iterations=cfg.ransac.num_iterations,
                )
            elif args.method == "svd":
                estimated_transform = weighted_procrustes(src_corr_points, tgt_corr_points, corr_scores)
            else:
                raise ValueError(f"Unsupported registration method: {args.method}.")

            estimated_transforms.append(
                {
                    "test_pair": [tgt_frame, src_frame],
                    "num_fragments": num_fragments,
                    "transform": estimated_transform,
                }
            )

            if gt_indices[tgt_frame, src_frame] != -1:
                # evaluate transform (realignment error)
                gt_index = gt_indices[tgt_frame, src_frame]
                transform = gt_logs[gt_index]["transform"]
                covariance = gt_infos[gt_index]["covariance"]
                error = compute_transform_error(transform, covariance, estimated_transform)
                message += ", RMSE: {:.3f}".format(np.sqrt(error))
                accepted = error < cfg.eval.rmse_threshold**2
                registration_meter.update("scene_recall", float(accepted))
                if accepted:
                    rre, rte = isotropic_registration_error(transform, estimated_transform)
                    registration_meter.update("scene_rre", rre)
                    registration_meter.update("scene_rte", rte)
                    message += ", RRE: {:.3f}".format(rre)
                    message += ", RTE: {:.3f}".format(rte)

            # Evaluate re-alignment error
            # if tgt_frame + 1 < src_frame:
            #     evaluate transform (realignment error)
            #     src_points_f = data_dict['src_points_f']
            #     error = compute_realignment_error(src_points_f, transform, estimated_transform)
            #     message += ', r_RMSE: {:.3f}'.format(error)
            #     accepted = error < config.eval_rmse_threshold
            #     registration_meter.update('scene_recall', float(accepted))
            #     if accepted:
            #         rre, rte = isotropic_registration_error(transform, estimated_transform)
            #         registration_meter.update('scene_rre', rre)
            #         registration_meter.update('scene_rte', rte)
            #         message += ', r_RRE: {:.3f}, r_RTE: {:.3f}'.format(rre, rte)

            if args.verbose:
                logger.info(message)

        est_log = osp.join(cfg.exp.result_dir, benchmark, scene_name, "est.log")
        write_log_file(est_log, estimated_transforms)

        logger.info(f"Scene_name: {scene_name}")

        # 1. print correspondence evaluation results (one scene)
        # 1.1 fine level statistics
        recall = fine_matching_meter.mean("scene_recall")
        inlier_ratio = fine_matching_meter.mean("scene_inlier_ratio")
        overlap = fine_matching_meter.mean("scene_overlap")
        fine_matching_meter.update("recall", recall)
        fine_matching_meter.update("inlier_ratio", inlier_ratio)
        fine_matching_meter.update("overlap", overlap)
        scene_fine_matching_result_dict[scene_abbr] = {"recall": recall, "inlier_ratio": inlier_ratio}

        message = "  Correspondence"
        message += ", FMR: {:.3f}".format(recall)
        message += ", IR: {:.3f}".format(inlier_ratio)
        message += ", OV: {:.3f}".format(overlap)
        logger.info(message)

        # 2. print registration evaluation results (one scene)
        recall = registration_meter.mean("scene_recall")
        mean_rre = registration_meter.mean("scene_rre")
        mean_rte = registration_meter.mean("scene_rte")
        median_rre = registration_meter.median("scene_rre")
        median_rte = registration_meter.median("scene_rte")
        registration_meter.update("recall", recall)
        registration_meter.update("mean_rre", mean_rre)
        registration_meter.update("mean_rte", mean_rte)
        registration_meter.update("median_rre", median_rre)
        registration_meter.update("median_rte", median_rte)

        scene_registration_result_dict[scene_abbr] = {
            "recall": recall,
            "mean_rre": mean_rre,
            "mean_rte": mean_rte,
            "median_rre": median_rre,
            "median_rte": median_rte,
        }

        message = "  Registration"
        message += ", RR: {:.3f}".format(recall)
        message += ", mean_RRE: {:.3f}".format(mean_rre)
        message += ", mean_RTE: {:.3f}".format(mean_rte)
        message += ", median_RRE: {:.3f}".format(median_rre)
        message += ", median_RTE: {:.3f}".format(median_rte)
        logger.info(message)

    # 1. print correspondence evaluation results
    message = "  Matching"
    message += ", FMR: {:.3f}".format(fine_matching_meter.mean("recall"))
    message += ", IR: {:.3f}".format(fine_matching_meter.mean("inlier_ratio"))
    message += ", OV: {:.3f}".format(fine_matching_meter.mean("overlap"))
    message += ", std: {:.3f}".format(fine_matching_meter.std("recall"))
    logger.success(message)
    for scene_abbr, result_dict in scene_fine_matching_result_dict.items():
        message = "    {}".format(scene_abbr)
        message += ", FMR: {:.3f}".format(result_dict["recall"])
        message += ", IR: {:.3f}".format(result_dict["inlier_ratio"])
        logger.success(message)

    # 2. print registration evaluation results
    message = "  Registration"
    message += ", RR: {:.3f}".format(registration_meter.mean("recall"))
    message += ", mean_RRE: {:.3f}".format(registration_meter.mean("mean_rre"))
    message += ", mean_RTE: {:.3f}".format(registration_meter.mean("mean_rte"))
    message += ", median_RRE: {:.3f}".format(registration_meter.mean("median_rre"))
    message += ", median_RTE: {:.3f}".format(registration_meter.mean("median_rte"))
    logger.success(message)
    for scene_abbr, result_dict in scene_registration_result_dict.items():
        message = "    {}".format(scene_abbr)
        message += ", RR: {:.3f}".format(result_dict["recall"])
        message += ", mean_RRE: {:.3f}".format(result_dict["mean_rre"])
        message += ", mean_RTE: {:.3f}".format(result_dict["mean_rte"])
        message += ", median_RRE: {:.3f}".format(result_dict["median_rre"])
        message += ", median_RTE: {:.3f}".format(result_dict["median_rte"])
        logger.success(message)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.exp.log_dir, "eval-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
    logger = get_logger(log_file=log_file)

    message = "Configs:\n" + json.dumps(cfg, indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, logger)


if __name__ == "__main__":
    main()
