import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import BasicConvResBlock, ConvBlock, KPConvBlock, KPResidualBlock, UnaryBlockPackMode
from vision3d.ops import knn_interpolate_pack_mode, render


class ImageBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.stem = ConvBlock(
            input_dim,
            hidden_dim,
            kernel_size=7,
            padding=3,
            stride=2,
            conv_cfg="Conv2d",
            norm_cfg="GroupNorm",
            act_cfg="LeakyReLU",
        )

        self.block1 = BasicConvResBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, conv_cfg="Conv2d")

        self.block2 = BasicConvResBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, conv_cfg="Conv2d")

        self.out_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, image):
        image_h = image.shape[2]
        image_w = image.shape[3]
        feats = self.stem(image)
        feats = self.block1(feats)
        feats = self.block2(feats)
        feats = self.out_proj(feats)
        feats = F.interpolate(feats, (image_h, image_w), mode="bilinear", align_corners=True)
        return feats


class PointBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, kernel_size, init_radius, init_sigma):
        super().__init__()

        self.encoder1_1 = KPConvBlock(input_dim, hidden_dim, kernel_size, init_radius, init_sigma)
        self.encoder1_2 = KPResidualBlock(hidden_dim, hidden_dim, kernel_size, init_radius, init_sigma)

        self.encoder2_1 = KPResidualBlock(hidden_dim, hidden_dim, kernel_size, init_radius, init_sigma, strided=True)
        self.encoder2_2 = KPResidualBlock(hidden_dim, hidden_dim, kernel_size, init_radius * 2, init_sigma * 2)

        self.decoder1 = UnaryBlockPackMode(hidden_dim * 2, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, feats, data_dict):
        points_list = data_dict["points"]
        neighbors_list = data_dict["neighbors"]
        subsampling_list = data_dict["subsampling"]
        upsampling_list = data_dict["upsampling"]

        feats_s1 = feats
        feats_s1 = self.encoder1_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder1_2(points_list[0], points_list[0], feats_s1, neighbors_list[0])

        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        latent_s2 = feats_s2

        latent_s1 = knn_interpolate_pack_mode(points_list[0], points_list[1], latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)

        latent_s1 = self.out_proj(latent_s1)

        return latent_s1


def collect_img_feats(all_img_feats, all_pcd_pixels, all_pcd_masks):
    assert all_img_feats.shape[0] == all_pcd_pixels.shape[0]

    num_channels = all_img_feats.shape[3]
    num_points = all_pcd_pixels.shape[1]

    pcd_counts = torch.zeros(size=(num_points,)).cuda()
    pcd_feats = torch.zeros(size=(num_points, num_channels)).cuda()

    all_pcd_h_indices = all_pcd_pixels[..., 0]
    all_pcd_w_indices = all_pcd_pixels[..., 1]
    image_indices, point_indices = torch.nonzero(all_pcd_masks, as_tuple=True)
    h_indices = all_pcd_h_indices[image_indices, point_indices]
    w_indices = all_pcd_w_indices[image_indices, point_indices]

    ones = torch.ones_like(point_indices, dtype=torch.float)
    pcd_counts.scatter_add_(dim=0, index=point_indices, src=ones)

    sel_img_feats = all_img_feats[image_indices, h_indices, w_indices]
    point_indices = point_indices.unsqueeze(1).expand_as(sel_img_feats)
    pcd_feats.scatter_add_(dim=0, index=point_indices, src=sel_img_feats)

    pcd_feats = pcd_feats / torch.clamp(pcd_counts.unsqueeze(1), min=1e-10)

    return pcd_feats


def collect_pcd_feats(pcd_feats, all_pcd_pixels, all_pcd_masks, image_h, image_w):
    assert pcd_feats.shape[0] == all_pcd_pixels.shape[1]

    num_channels = pcd_feats.shape[1]
    num_images = all_pcd_pixels.shape[0]

    img_feats = torch.zeros(size=(num_images * image_h * image_w, num_channels)).cuda()
    img_counts = torch.zeros(size=(num_images * image_h * image_w,)).cuda()

    all_pcd_h_indices = all_pcd_pixels[..., 0]
    all_pcd_w_indices = all_pcd_pixels[..., 1]
    image_indices, point_indices = torch.nonzero(all_pcd_masks, as_tuple=True)
    h_indices = all_pcd_h_indices[image_indices, point_indices]
    w_indices = all_pcd_w_indices[image_indices, point_indices]
    flat_indices = image_indices * image_h * image_w + h_indices * image_w + w_indices

    ones = torch.ones_like(flat_indices, dtype=torch.float)
    img_counts.scatter_add_(dim=0, index=flat_indices, src=ones)

    sel_pcd_feats = pcd_feats[point_indices]
    flat_indices = flat_indices.unsqueeze(1).expand_as(sel_pcd_feats)
    img_feats.scatter_add_(dim=0, index=flat_indices, src=sel_pcd_feats)

    img_feats = img_feats / torch.clamp(img_counts.unsqueeze(1), min=1e-10)
    img_feats = img_feats.view(num_images, image_h, image_w, num_channels)

    return img_feats


def compute_masks(pcd_pixels, image_h, image_w):
    h_indices = pcd_pixels[..., 0]
    w_indices = pcd_pixels[..., 1]
    h_masks = torch.logical_and(torch.ge(h_indices, 0), torch.lt(h_indices, image_h))
    w_masks = torch.logical_and(torch.ge(w_indices, 0), torch.lt(w_indices, image_w))
    masks = torch.logical_and(h_masks, w_masks)
    return masks


class FusionModule(nn.Module):
    def __init__(self, output_dim, hidden_dim, kernel_size, init_radius, init_sigma):
        super().__init__()

        self.img_net_1 = ImageBackbone(3, output_dim // 2, hidden_dim)
        self.pcd_net_1 = PointBackbone(1, output_dim // 2, hidden_dim, kernel_size, init_radius, init_sigma)
        self.img_net_2 = ImageBackbone(output_dim, output_dim // 2, hidden_dim)
        self.pcd_net_2 = PointBackbone(output_dim, output_dim // 2, hidden_dim, kernel_size, init_radius, init_sigma)

        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(
        self, src_images, tgt_images, src_points, tgt_points, src_transforms, tgt_transforms, intrinsics, data_dict
    ):
        assert src_images.shape[0] == tgt_images.shape[0]
        assert src_images.shape[1] == tgt_images.shape[1]
        assert src_images.shape[2] == tgt_images.shape[2]
        assert src_images.shape[0] == src_transforms.shape[0]
        assert tgt_images.shape[0] == tgt_transforms.shape[0]

        # cm = get_context_manager()

        num_images = src_images.shape[0]
        image_h = src_images.shape[1]
        image_w = src_images.shape[2]
        src_length = src_points.shape[0]

        # render points to pixels
        all_src_pixels = render(src_points.unsqueeze(0), intrinsics.unsqueeze(0), extrinsics=src_transforms)
        all_tgt_pixels = render(tgt_points.unsqueeze(0), intrinsics.unsqueeze(0), extrinsics=tgt_transforms)
        all_src_masks = compute_masks(all_src_pixels, image_h, image_w)
        all_tgt_masks = compute_masks(all_tgt_pixels, image_h, image_w)

        # img
        img_feats_s1 = torch.cat([src_images, tgt_images], dim=0)  # (4, H, W, 3)
        img_feats_s1 = img_feats_s1.permute(0, 3, 1, 2).contiguous()  # (4, 3, H, W)
        img_feats_s1 = self.img_net_1(img_feats_s1)  # (4, C, H, W)
        img_feats_s1 = img_feats_s1.permute(0, 2, 3, 1).contiguous()  # (4, H, W, C)
        src_img_feats_s1 = img_feats_s1[:num_images]
        tgt_img_feats_s1 = img_feats_s1[num_images:]

        # pcd
        pcd_feats_s1 = data_dict["feats"].detach()
        pcd_feats_s1 = self.pcd_net_1(pcd_feats_s1, data_dict)

        # PRINT BEGIN
        src_pcd_feats_s1 = pcd_feats_s1[:src_length]
        tgt_pcd_feats_s1 = pcd_feats_s1[src_length:]
        # cm.register("src_pcd_feats_s1", src_pcd_feats_s1)
        # cm.register("tgt_pcd_feats_s1", tgt_pcd_feats_s1)
        # PRINT END

        # pcd+img -> pcd
        src_pcd_feats_from_img = collect_img_feats(src_img_feats_s1, all_src_pixels, all_src_masks)
        tgt_pcd_feats_from_img = collect_img_feats(tgt_img_feats_s1, all_tgt_pixels, all_tgt_masks)
        pcd_feats_from_img = torch.cat([src_pcd_feats_from_img, tgt_pcd_feats_from_img], dim=0)
        pcd_feats_s2 = torch.cat([pcd_feats_s1, pcd_feats_from_img], dim=1)  # (N, 2*C)

        # PRINT BEGIN
        src_pcd_feats_s2 = pcd_feats_s2[:src_length]
        tgt_pcd_feats_s2 = pcd_feats_s2[src_length:]
        # cm.register("src_pcd_feats_s2", src_pcd_feats_s2)
        # cm.register("tgt_pcd_feats_s2", tgt_pcd_feats_s2)
        # PRINT END

        pcd_feats_s2 = self.pcd_net_2(pcd_feats_s2, data_dict)
        src_pcd_feats_s2 = pcd_feats_s2[:src_length]
        tgt_pcd_feats_s2 = pcd_feats_s2[src_length:]

        # img+pcd -> img
        src_img_feats_from_pcd = collect_pcd_feats(src_pcd_feats_s2, all_src_pixels, all_src_masks, image_h, image_w)
        tgt_img_feats_from_pcd = collect_pcd_feats(tgt_pcd_feats_s2, all_tgt_pixels, all_tgt_masks, image_h, image_w)
        img_feats_from_pcd = torch.cat([src_img_feats_from_pcd, tgt_img_feats_from_pcd], dim=0)  # (4, H, W, C)
        img_feats_s2 = torch.cat([img_feats_s1, img_feats_from_pcd], dim=3)  # (4, H, W, 2*C)
        img_feats_s2 = img_feats_s2.permute(0, 3, 1, 2).contiguous()  # (4, 2*C, H, W)
        img_feats_s2 = self.img_net_2(img_feats_s2)
        img_feats_s2 = img_feats_s2.permute(0, 2, 3, 1).contiguous()  # (4, H, W, C)
        src_img_feats_s2 = img_feats_s2[:num_images]
        tgt_img_feats_s2 = img_feats_s2[num_images:]

        # pcd+img -> pcd
        src_pcd_feats_from_img = collect_img_feats(src_img_feats_s2, all_src_pixels, all_src_masks)
        tgt_pcd_feats_from_img = collect_img_feats(tgt_img_feats_s2, all_tgt_pixels, all_tgt_masks)
        pcd_feats_from_img = torch.cat([src_pcd_feats_from_img, tgt_pcd_feats_from_img], dim=0)
        pcd_feats_final = torch.cat([pcd_feats_s2, pcd_feats_from_img], dim=1)  # (N, 2*C)
        pcd_feats_final = self.out_proj(pcd_feats_final)
        src_pcd_feats = pcd_feats_final[:src_length]
        tgt_pcd_feats = pcd_feats_final[src_length:]

        # PRINT BEGIN
        # src_pcd_feats_final = pcd_feats_final[:src_length]
        # tgt_pcd_feats_final = pcd_feats_final[src_length:]
        # cm.register("src_pcd_feats_final", src_pcd_feats_final)
        # cm.register("tgt_pcd_feats_final", tgt_pcd_feats_final)
        # PRINT END

        return src_pcd_feats, tgt_pcd_feats
