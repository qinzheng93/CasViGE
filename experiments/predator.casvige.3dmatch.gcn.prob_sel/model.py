import torch
import torch.nn as nn
from torch.nn import functional as F

from vision3d.models.predator import GCN

# isort: split
from backbone import PointDecoder, PointEncoder
from casvige import FusionModule


class OverlapPredator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Parameters
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

        self.fusion = FusionModule(
            cfg.model.fusion.output_dim,
            cfg.model.fusion.hidden_dim,
            cfg.model.backbone.kernel_size,
            cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_radius,
            cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_sigma,
        )

        # encoder
        self.encoder = PointEncoder(
            cfg.model.backbone.input_dim,
            cfg.model.backbone.init_dim,
            cfg.model.backbone.kernel_size,
            cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_radius,
            cfg.model.backbone.base_voxel_size * cfg.model.backbone.kpconv_sigma,
        )

        # decoder
        self.decoder = PointDecoder(
            cfg.model.gcn.hidden_dim + 2,
            cfg.model.backbone.output_dim + 2,
            cfg.model.backbone.init_dim,
        )

        # bottleneck layer and GNN part
        self.in_proj = nn.Conv1d(cfg.model.backbone.init_dim * 16, cfg.model.gcn.hidden_dim, kernel_size=1)
        self.gcn = GCN(cfg.model.gcn.num_heads, cfg.model.gcn.hidden_dim, cfg.model.gcn.k, cfg.model.gcn.blocks)
        self.out_proj = nn.Conv1d(cfg.model.gcn.hidden_dim, cfg.model.gcn.hidden_dim, kernel_size=1)
        self.score_proj = nn.Conv1d(cfg.model.gcn.hidden_dim, 1, kernel_size=1)

    def forward(self, data_dict):
        output_dict = {}

        # Unpack data
        src_length_c = data_dict["lengths"][-1][0].item()
        src_length_f = data_dict["lengths"][0][0].item()
        points_c = data_dict["points"][-1].detach()
        points_f = data_dict["points"][0].detach()

        src_points_c = points_c[:src_length_c]
        tgt_points_c = points_c[src_length_c:]
        src_points_f = points_f[:src_length_f]
        tgt_points_f = points_f[src_length_f:]

        output_dict["src_points"] = src_points_f
        output_dict["tgt_points"] = tgt_points_f

        src_images = data_dict["src_images"].detach()
        tgt_images = data_dict["tgt_images"].detach()
        src_transforms = data_dict["src_transforms"].detach()
        tgt_transforms = data_dict["tgt_transforms"].detach()
        intrinsics = data_dict["intrinsics"].detach()

        # 1. 2D-3D Fusion
        src_feats, tgt_feats = self.fusion(
            src_images,
            tgt_images,
            src_points_f,
            tgt_points_f,
            src_transforms,
            tgt_transforms,
            intrinsics,
            data_dict,
        )

        # 1. KPFCNN Encoder
        feats = torch.cat([src_feats, tgt_feats], dim=0)
        feats_list = self.encoder(feats, data_dict)

        # 2. GCN
        feats_c = feats_list.pop(-1)
        src_feats_c = feats_c[:src_length_c]
        tgt_feats_c = feats_c[src_length_c:]
        src_feats_c = src_feats_c.transpose(0, 1).unsqueeze(0)  # (N, C) -> (1, C, N)
        tgt_feats_c = tgt_feats_c.transpose(0, 1).unsqueeze(0)  # (N, C) -> (1, C, N)
        src_points_c = src_points_c.transpose(0, 1).unsqueeze(0)
        tgt_points_c = tgt_points_c.transpose(0, 1).unsqueeze(0)
        src_feats_c = self.in_proj(src_feats_c)
        tgt_feats_c = self.in_proj(tgt_feats_c)

        src_feats_c, tgt_feats_c = self.gcn(src_points_c, tgt_points_c, src_feats_c, tgt_feats_c)

        src_feats_c = self.out_proj(src_feats_c)
        tgt_feats_c = self.out_proj(tgt_feats_c)
        src_scores_c = self.score_proj(src_feats_c)
        tgt_scores_c = self.score_proj(tgt_feats_c)

        src_feats_c = src_feats_c.squeeze(0).transpose(0, 1)
        tgt_feats_c = tgt_feats_c.squeeze(0).transpose(0, 1)
        src_scores_c = src_scores_c.squeeze(0).transpose(0, 1)
        tgt_scores_c = tgt_scores_c.squeeze(0).transpose(0, 1)

        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=0)
        scores_c = torch.cat([src_scores_c, tgt_scores_c], dim=0)

        # 3. Cross Scores
        src_feats_c_norm = F.normalize(src_feats_c, p=2, dim=1)
        tgt_feats_c_norm = F.normalize(tgt_feats_c, p=2, dim=1)
        temperature = torch.exp(self.epsilon) + 0.03
        attention_scores = torch.einsum("nc,mc->nm", src_feats_c_norm, tgt_feats_c_norm) / temperature
        cross_src_scores_c = torch.matmul(F.softmax(attention_scores, dim=1), tgt_scores_c)
        cross_tgt_scores_c = torch.matmul(F.softmax(attention_scores.transpose(0, 1), dim=1), src_scores_c)
        cross_scores_c = torch.cat([cross_src_scores_c, cross_tgt_scores_c], dim=0)
        feats_c = torch.cat([scores_c, cross_scores_c, feats_c], dim=1)

        # print(feats_c.shape)

        feats_list.append(feats_c)

        # 4. KPFCNN Decoder
        outputs = self.decoder(feats_list, data_dict)

        # 5. Post Computation
        feats = outputs[:, :-2]
        overlap_scores = outputs[:, -2]
        saliency_scores = outputs[:, -1]

        feats = F.normalize(feats, p=2, dim=1)
        overlap_scores = torch.sigmoid(overlap_scores)
        saliency_scores = torch.sigmoid(saliency_scores)
        scores = overlap_scores * saliency_scores

        output_dict["src_feats"] = feats[:src_length_f]
        output_dict["tgt_feats"] = feats[src_length_f:]
        output_dict["src_overlap_scores"] = overlap_scores[:src_length_f]
        output_dict["tgt_overlap_scores"] = overlap_scores[src_length_f:]
        output_dict["src_saliency_scores"] = saliency_scores[:src_length_f]
        output_dict["tgt_saliency_scores"] = saliency_scores[src_length_f:]
        output_dict["src_scores"] = scores[:src_length_f]
        output_dict["tgt_scores"] = scores[src_length_f:]

        return output_dict


def create_model(cfg):
    model = OverlapPredator(cfg)
    return model


def run_test():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model)


if __name__ == "__main__":
    run_test()
