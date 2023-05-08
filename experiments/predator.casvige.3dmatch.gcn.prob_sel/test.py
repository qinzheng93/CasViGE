import os.path as osp

import numpy as np

from vision3d.engine import SingleTester
from vision3d.utils.io import ensure_dir
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args, get_default_parser
from vision3d.utils.profiling import profile_cpu_runtime
from vision3d.utils.tensor import tensor_to_array

# isort: split
from config import make_cfg
from dataset import test_data_loader
from loss import EvalFunction
from model import create_model


def add_custom_args():
    parser = get_default_parser()
    parser.add_argument_group("experiment", "experiment arguments")
    parser.add_argument("--benchmark", choices=["3DMatch", "3DLoMatch", "extra", "val"], help="test benchmark")


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader create"):
            data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.eval_func = EvalFunction(cfg)

        # preparation
        self.output_dir = osp.join(cfg.exp.cache_dir, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)
        return result_dict

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        src_frame = data_dict["src_frame"]
        tgt_frame = data_dict["tgt_frame"]

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f"{tgt_frame}_{src_frame}.npz")
        np.savez_compressed(
            file_name,
            src_points=tensor_to_array(output_dict["src_points"]),
            tgt_points=tensor_to_array(output_dict["tgt_points"]),
            src_scores=tensor_to_array(output_dict["src_scores"]),
            tgt_scores=tensor_to_array(output_dict["tgt_scores"]),
            src_feats=tensor_to_array(output_dict["src_feats"]),
            tgt_feats=tensor_to_array(output_dict["tgt_feats"]),
            src_overlap_scores=tensor_to_array(output_dict["src_overlap_scores"]),
            tgt_overlap_scores=tensor_to_array(output_dict["tgt_overlap_scores"]),
            src_saliency_scores=tensor_to_array(output_dict["src_saliency_scores"]),
            tgt_saliency_scores=tensor_to_array(output_dict["tgt_saliency_scores"]),
            transform=tensor_to_array(data_dict["transform"]),
            overlap=data_dict["overlap"],
        )


def main():
    add_tester_args()
    add_custom_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
