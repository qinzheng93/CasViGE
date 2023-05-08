from vision3d.engine import EpochBasedTrainer
from vision3d.utils.optimizer import build_optimizer, build_scheduler
from vision3d.utils.parser import add_trainer_args
from vision3d.utils.profiling import profile_cpu_runtime

# isort: split
from config import make_cfg
from dataset import train_valid_data_loader
from loss import EvalFunction, LossFunction
from model import create_model


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader create"):
            train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg)
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(train_loader, val_loader)

        # model
        model = create_model(cfg)
        model = self.register_model(model)

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = LossFunction(cfg)
        self.eval_func = EvalFunction(cfg)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def after_val_epoch(self, epoch, summary_dict):
        if summary_dict["d_recall"] > 0.3:
            self.loss_func.saliency_loss.weight = 1.0


def main():
    add_trainer_args()
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
