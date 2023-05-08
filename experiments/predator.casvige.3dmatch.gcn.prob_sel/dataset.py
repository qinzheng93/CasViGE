from vision3d.datasets.registration import ThreeDMatchRgbPairDataset
from vision3d.utils.collate import GraphPyramidRegistrationCollateFnPackMode
from vision3d.utils.dataloader import build_dataloader, calibrate_neighbors_pack_mode


def train_valid_data_loader(cfg):
    train_dataset = ThreeDMatchRgbPairDataset(
        cfg.data.dataset_dir,
        "train",
        max_points=cfg.train.max_points,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        use_weak_augmentation=cfg.train.use_weak_augmentation,
        return_corr_indices=True,
        matching_radius=cfg.data.matching_radius,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
        image_frames=cfg.train.image_frames,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramidRegistrationCollateFnPackMode,
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_radius,
    )

    collate_fn = GraphPyramidRegistrationCollateFnPackMode(
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_radius,
        neighbor_limits,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataset = ThreeDMatchRgbPairDataset(
        cfg.data.dataset_dir,
        "val",
        max_points=cfg.test.max_points,
        use_augmentation=False,
        return_corr_indices=True,
        matching_radius=cfg.data.matching_radius,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
        image_frames=cfg.train.image_frames,
    )

    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):
    train_dataset = ThreeDMatchRgbPairDataset(
        cfg.data.dataset_dir,
        "train",
        max_points=cfg.train.max_points,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
        return_corr_indices=True,
        matching_radius=cfg.data.matching_radius,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
        image_frames=cfg.train.image_frames,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramidRegistrationCollateFnPackMode,
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_radius,
    )

    collate_fn = GraphPyramidRegistrationCollateFnPackMode(
        cfg.model.backbone.num_stages,
        cfg.model.backbone.base_voxel_size,
        cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_radius,
        neighbor_limits,
    )

    test_dataset = ThreeDMatchRgbPairDataset(
        cfg.data.dataset_dir,
        benchmark,
        max_points=cfg.test.max_points,
        use_augmentation=False,
        return_corr_indices=True,
        matching_radius=cfg.data.matching_radius,
        image_h=cfg.data.image_h,
        image_w=cfg.data.image_w,
        image_frames=cfg.test.image_frames,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return test_loader, neighbor_limits
