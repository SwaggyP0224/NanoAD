import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from Feature_Aggregated import Feature_Aggregator
from model import Discriminator, Projection, PatchMaker
from NanoAD_lib.Loss import FocalLoss, DiceFocalLoss


class LearnableResidual(nn.Module):
    def __init__(self, feature_dim, num_prototypes=16):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, feature_dim))
        nn.init.xavier_normal_(self.prototypes)
        self.is_initialized = False

    def init_prototypes(self, dataloader, teacher, device, max_samples=2000):
        print(f"Initializing prototypes via K-Means (K={self.num_prototypes})")
        teacher.eval()
        features_list = []
        sample_count = 0

        from sklearn.cluster import KMeans

        with torch.no_grad():
            for batch in dataloader:
                img = batch['image'].to(device)
                _, feats, _ = teacher(img)

                feats_flat = feats.reshape(-1, self.feature_dim).cpu()
                features_list.append(feats_flat)

                sample_count += feats_flat.shape[0]
                if sample_count >= max_samples:
                    break

        all_feats = torch.cat(features_list, dim=0).numpy()

        print(f"Running K-Means on {all_feats.shape[0]} feature vectors")
        kmeans = KMeans(n_clusters=self.num_prototypes, n_init=10, random_state=2)
        kmeans.fit(all_feats)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        self.prototypes.data.copy_(centers)
        self.is_initialized = True
        print("Prototypes initialized successfully")

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, self.feature_dim)
        else:
            x_flat = x

        dists = torch.cdist(x_flat, self.prototypes)

        min_vals, min_indices = torch.min(dists, dim=1)

        nearest_protos = self.prototypes[min_indices]

        residual = x_flat - nearest_protos

        if len(original_shape) == 3:
            residual = residual.reshape(original_shape)

        return residual, min_vals


class NanoAD(nn.Module):
    def __init__(self, c, Feature_Aggregator_params, bottleneck, student):
        super().__init__()
        self._class_ = getattr(c, '_class_', 'default')
        self.c = c
        self.t = Feature_Aggregator(**Feature_Aggregator_params)
        self.target_embed_dimension = self.t.target_embed_dimension
        self.type = 'train'

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, c.image_size, c.image_size).to(Feature_Aggregator_params['device'])
            _, _, patch_shape = self.t(dummy_input)
            self.num_patches = patch_shape[0] * patch_shape[1]
            print(f"Computed Patch Info: Grid={patch_shape}, Total Patches={self.num_patches}")

        self.use_residual_module = (c.svdd_type in ['residual', 'multi_center']) and (not c.no_residual)

        if self.use_residual_module:
            num_protos = getattr(c, 'num_prototypes', 16)
            print(f"Initializing Learnable Residual Module with {num_protos} prototypes")
            self.residual_module = LearnableResidual(
                feature_dim=self.target_embed_dimension,
                num_prototypes=num_protos
            )
        else:
            print("Ablation: Residual Module DISABLED (Identity mapping)")
            self.residual_module = nn.Identity()

        from model import BN, Adapter

        if hasattr(c, 'ablation_no_bn_student') and c.ablation_no_bn_student:
            self.bn = nn.Identity()
            self.BN_class = type(None)
        else:
            self.bn = BN(bottleneck)
            self.BN_class = BN

            if c.no_adapter or (hasattr(c, 'ablation_no_bn_student') and c.ablation_no_bn_student):
                self.s = nn.Identity()
                self.Student_class = type(None)
            else:
                no_pe = getattr(c, 'no_pe', False)
                attn_mode = getattr(c, 'attention_mode', 'sha')

                self.s = Adapter(student, num_patches=self.num_patches,
                                 embed_dim=self.target_embed_dimension,
                                 use_se=True,
                                 no_pe=no_pe,
                                 attention_mode=attn_mode)
                self.Student_class = Adapter

        if c.use_dfs:
            if not hasattr(c, 'dfs_channels'):
                raise AttributeError("Config 'c' needs 'dfs_channels' when use_dfs is True.")
            pass
        else:
            self.dfs = None

        if c.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, c.pre_proj)
        else:
            self.pre_projection = nn.Identity()

        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=c.dsc_layers, hidden=c.dsc_hidden)
        self.focal_loss = DiceFocalLoss(lambda_dice=1.0, lambda_focal=1.0)
        self.patch_maker = PatchMaker(patchsize=c.patchsize, stride=1)
        self.register_buffer('center', torch.zeros(1, self.target_embed_dimension))

    def train_or_eval(self, type='train'):
        self.type = type
        self.t.train_eval(type)

        modules = [self.bn, self.s, self.pre_projection, self.discriminator, self.residual_module]
        if self.dfs: modules.append(self.dfs)

        for m in modules:
            if hasattr(m, 'train_eval'):
                m.train_eval(type)
            else:
                m.train() if type == 'train' else m.eval()
        return self

    def forward(self, x, x_aug=None):
        dfs_weights = None

        if self.type == 'train':
            combined_x = torch.cat([x, x_aug], dim=0)
            multi_level_features, aggregated_features, patch_shape = self.t(combined_x)

            features_to_process = aggregated_features
            if self.dfs:
                weighted_features, dfs_weights = self.dfs(multi_level_features, multi_level_features)
                features_to_process = self.t.aggregate_features(weighted_features)

            if self.c.svdd_type == 'residual' and self.use_residual_module:
                res_output = self.residual_module(features_to_process)
                residual_features, dist_vals = res_output if isinstance(res_output, tuple) else (res_output, None)
                forward_features = residual_features
            elif self.c.svdd_type == 'multi_center' and self.use_residual_module:
                res_output = self.residual_module(features_to_process)
                _, dist_vals = res_output if isinstance(res_output, tuple) else (None, None)
                forward_features = features_to_process
            else:
                forward_features = features_to_process
                dist_vals = None

            final_features = self.pre_projection(self.s(self.bn(forward_features)))

            true_feats, fake_feats = torch.chunk(final_features, 2, dim=0)

            svdd_loss = torch.tensor(0.0, device=x.device)

            if self.c.svdd_type in ['residual', 'multi_center']:
                if dist_vals is not None:
                    true_dists, _ = torch.chunk(dist_vals, 2, dim=0)
                    svdd_loss = torch.mean(true_dists)
                else:
                    svdd_loss = torch.mean(torch.sum((true_feats - self.center) ** 2, dim=1))

            return true_feats, fake_feats, dfs_weights, svdd_loss

        else:
            multi_level_features, aggregated_features, patch_shape = self.t(x)
            features_to_process = aggregated_features
            if self.dfs:
                weighted_features, _ = self.dfs(multi_level_features, multi_level_features)
                features_to_process = self.t.aggregate_features(weighted_features)

            if self.c.svdd_type == 'residual' and self.use_residual_module:
                res_output = self.residual_module(features_to_process)
                forward_features = res_output[0] if isinstance(res_output, tuple) else res_output
            else:
                forward_features = features_to_process

            final_features = self.pre_projection(self.s(self.bn(forward_features)))
            patch_scores_raw = self.discriminator(final_features)
            pixel_scores = patch_scores_raw.reshape(x.shape[0], patch_shape[0], patch_shape[1])
            scores_flat = pixel_scores.view(x.shape[0], -1)
            image_scores = self.patch_maker.score(scores_flat)

            return image_scores, pixel_scores
