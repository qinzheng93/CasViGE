import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from vision3d.utils.io import ensure_dir

_C = edict()

# exp
_C.exp = edict()
_C.exp.name = osp.basename(osp.dirname(osp.realpath(__file__)))
_C.exp.working_dir = osp.dirname(osp.realpath(__file__))
_C.exp.output_dir = osp.join("..", "..", "outputs", _C.exp.name)
_C.exp.checkpoint_dir = osp.join(_C.exp.output_dir, "checkpoints")
_C.exp.log_dir = osp.join(_C.exp.output_dir, "logs")
_C.exp.event_dir = osp.join(_C.exp.output_dir, "events")
_C.exp.cache_dir = osp.join(_C.exp.output_dir, "cache")
_C.exp.result_dir = osp.join(_C.exp.output_dir, "results")
_C.exp.seed = 7351

ensure_dir(_C.exp.output_dir)
ensure_dir(_C.exp.checkpoint_dir)
ensure_dir(_C.exp.log_dir)
ensure_dir(_C.exp.event_dir)
ensure_dir(_C.exp.cache_dir)
ensure_dir(_C.exp.result_dir)

# data
_C.data = edict()
_C.data.dataset_dir = osp.join("..", "..", "data", "3DMatch")
_C.data.matching_radius = 0.0375
_C.data.image_h = 240
_C.data.image_w = 320

# train
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.max_points = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0
_C.train.use_weak_augmentation = False
_C.train.image_frames = (0, 4)

# test
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.max_points = None
_C.test.image_frames = (0, 2, 3, 4)

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2
_C.eval.rre_threshold = 15.0
_C.eval.rte_threshold = 0.3
_C.eval.num_keypoints = 1000

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 50000

# trainer
_C.trainer = edict()
_C.trainer.max_epoch = 40
_C.trainer.grad_acc_steps = 1

# optim
_C.optimizer = edict()
_C.optimizer.type = "Adam"
_C.optimizer.lr = 1e-4
_C.optimizer.weight_decay = 1e-6

# scheduler
_C.scheduler = edict()
_C.scheduler.type = "Step"
_C.scheduler.gamma = 0.95
_C.scheduler.step_size = 1

# model - KPFCNN
_C.model = edict()

_C.model.fusion = edict()
_C.model.fusion.output_dim = 64
_C.model.fusion.hidden_dim = 64

_C.model.backbone = edict()
_C.model.backbone.num_stages = 4
_C.model.backbone.base_voxel_size = 0.025
_C.model.backbone.kernel_size = 15
_C.model.backbone.kpconv_radius = 2.5
_C.model.backbone.kpconv_sigma = 2.0
_C.model.backbone.input_dim = 64
_C.model.backbone.init_dim = 64
_C.model.backbone.output_dim = 32

# model - GCN
_C.model.gcn = edict()
_C.model.gcn.hidden_dim = 256
_C.model.gcn.k = 10
_C.model.gcn.num_heads = 4
_C.model.gcn.blocks = ["self", "cross", "self"]

# loss
_C.loss = edict()

# loss - circle loss
_C.loss.circle_loss = edict()
_C.loss.circle_loss.positive_margin = 0.1
_C.loss.circle_loss.negative_margin = 1.4
_C.loss.circle_loss.positive_optimal = 0.1
_C.loss.circle_loss.negative_optimal = 1.4
_C.loss.circle_loss.log_scale = 24
_C.loss.circle_loss.max_correspondences = 256
_C.loss.circle_loss.positive_radius = 0.0375
_C.loss.circle_loss.negative_radius = 0.1
_C.loss.circle_loss.weight = 1.0

# loss - overlap loss
_C.loss.overlap_loss = edict()
_C.loss.overlap_loss.weight = 1.0

# loss - saliency loss
_C.loss.saliency_loss = edict()
_C.loss.saliency_loss.positive_radius = 0.05
_C.loss.saliency_loss.weight = 0.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
