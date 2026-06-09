import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import common
from model import PatchMaker
import numpy as np


class Feature_Aggregator(nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device, input_shape,
                 pretrain_embed_dimension, target_embed_dimension, patchsize=3, patchstride=1):
        super(Feature_Aggregator, self).__init__()
        self.device = device
        self.layers_to_extract_from = layers_to_extract_from
        self.target_embed_dimension = target_embed_dimension
        self.feature_aggregator = common.NetworkFeatureAggregator(backbone, self.layers_to_extract_from, self.device)

        feature_dimensions = self.feature_aggregator.feature_dimensions(input_shape)
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        total_input_dim = feature_dimensions

        self.preprocessing = common.Preprocessing(total_input_dim, pretrain_embed_dimension)
        self.preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)

    def aggregate_features(self, multi_level_features):
        ref_shape = multi_level_features[0].shape[-2:]
        aligned_features = [multi_level_features[0]]
        for i in range(1, len(multi_level_features)):
            aligned_feat = F.interpolate(multi_level_features[i], size=ref_shape, mode='bilinear', align_corners=False)
            aligned_features.append(aligned_feat)

        features_to_process = aligned_features

        patch_results = [self.patch_maker.patchify(features) for features in features_to_process]
        features_to_process_b_l_d = [res[0] for res in patch_results]

        features_to_process_bl_d = [f.contiguous().view(-1, f.shape[-1]) for f in features_to_process_b_l_d]

        preprocessed_features = self.preprocessing(features_to_process_bl_d)
        aggregated_features = self.preadapt_aggregator(preprocessed_features)

        return aggregated_features

    def forward(self, images):
        features_dict = self.feature_aggregator(images)
        multi_level_features = [features_dict[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(multi_level_features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                side = int(math.sqrt(L))
                multi_level_features[i] = feat.view(B, side, side, C).permute(0, 3, 1, 2)

        ref_shape = multi_level_features[0].shape[-2:]
        aligned_features = [multi_level_features[0]]
        for i in range(1, len(multi_level_features)):
            aligned_feat = F.interpolate(multi_level_features[i], size=ref_shape, mode='bilinear', align_corners=False)
            aligned_features.append(aligned_feat)

        aggregated_features = self.aggregate_features(aligned_features)

        _, ref_patch_shape = self.patch_maker.patchify(aligned_features[0])

        return aligned_features, aggregated_features, ref_patch_shape

    def train_eval(self, type='train'):
        self.eval()
        return self
