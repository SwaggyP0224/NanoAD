import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from Feature_Aggregated import Feature_Aggregator
from NanoAD import NanoAD
from datasets import get_dataloaders_for_gnet
from test import test_model
from model import SelectiveHybridAttention


def train(c):
    print(f"Configuring training environment for NanoAD: {c._class_}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, test_dataloader = get_dataloaders_for_gnet(c)

    if c.backbone == 'resnet18':
        backbone = models.resnet18(weights='IMAGENET1K_V1').to(device)
    else:
        backbone = models.wide_resnet50_2(weights='IMAGENET1K_V1').to(device)

    Feature_Aggregator_params = {
        "backbone": backbone, "layers_to_extract_from": c.layers_to_extract, "device": device,
        "input_shape": (3, c.image_size, c.image_size), "pretrain_embed_dimension": c.pretrain_embed_dimension,
        "target_embed_dimension": c.target_embed_dimension, "patchsize": c.patchsize, "patchstride": 1
    }

    bottleneck_dim = Feature_Aggregator_params['target_embed_dimension']
    bottleneck_model = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim // 2), nn.ReLU(),
                                     nn.Linear(bottleneck_dim // 2, bottleneck_dim)).to(device)
    student_model = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim), nn.ReLU(),
                                  nn.Linear(bottleneck_dim, bottleneck_dim)).to(device)

    gnet_model = NanoAD(c=c, Feature_Aggregator_params=Feature_Aggregator_params, bottleneck=bottleneck_model,
                        student=student_model).to(device)

    proj_opt = torch.optim.Adam(gnet_model.pre_projection.parameters(), c.lr, weight_decay=1e-5)
    dsc_opt = torch.optim.AdamW(gnet_model.discriminator.parameters(), lr=c.lr * 2)

    params_to_optimize = list(gnet_model.bn.parameters()) + \
                         list(gnet_model.s.parameters()) + \
                         list(gnet_model.residual_module.parameters())

    if gnet_model.dfs is not None:
        params_to_optimize.extend(list(gnet_model.dfs.parameters()))

    main_opt = torch.optim.Adam(params_to_optimize, lr=c.lr)

    scheduler_proj = CosineAnnealingLR(proj_opt, T_max=c.epochs, eta_min=1e-6)
    scheduler_dsc = CosineAnnealingLR(dsc_opt, T_max=c.epochs, eta_min=1e-6)
    scheduler_main = CosineAnnealingLR(main_opt, T_max=c.epochs, eta_min=1e-6)

    should_run_kmeans = (not c.no_kmeans) and \
                        (not isinstance(gnet_model.residual_module, nn.Identity)) and \
                        (c.svdd_type in ['residual', 'multi_center'])

    if should_run_kmeans:
        print("\nExecuting K-Means Initialization for Prototypes")
        gnet_model.residual_module.init_prototypes(train_dataloader, gnet_model.t, device)
    else:
        print("\nSkipping K-Means Initialization (Disabled by args or not applicable)")

    print("Precomputing feature center for normal samples")
    gnet_model.train_or_eval('eval')
    all_image_features = []
    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Computing Center"):
            images = batch['image'].to(device)
            _, patch_features_raw, patch_shape = gnet_model.t(images)

            res_out = gnet_model.residual_module(patch_features_raw)
            if isinstance(res_out, tuple):
                residual_feats = res_out[0]
            else:
                residual_feats = res_out

            patch_features = gnet_model.pre_projection(gnet_model.s(gnet_model.bn(residual_feats)))

            num_patches = patch_shape[0] * patch_shape[1]
            image_features = torch.mean(patch_features.view(images.shape[0], num_patches, -1), dim=1)
            all_image_features.append(image_features)

    center_features = torch.cat(all_image_features, dim=0).mean(dim=0, keepdim=True)
    gnet_model.center.copy_(center_features)
    print(f"Feature center computed\n")

    print("Starting training loop")
    best_score = 0.0
    best_epoch = 0

    for epoch in range(c.epochs):
        gnet_model.train_or_eval('train')
        total_loss_epoch = 0
        total_svdd_epoch = 0.0

        avg_alpha_epoch = 0.0
        batch_count = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{c.epochs}")

        for batch in progress_bar:
            proj_opt.zero_grad()
            dsc_opt.zero_grad()
            main_opt.zero_grad()

            good_images = batch['image'].to(device)
            augmented_images = batch['aug'].to(device)
            mask_s = batch['mask_s'].to(device)

            if c.no_dtd:
                augmented_images = good_images.clone()
                mask_s = torch.zeros_like(mask_s)

            true_feats, fake_feats, dfs_weights, svdd_loss = gnet_model(good_images, x_aug=augmented_images)

            if not c.no_noise:
                noise = torch.normal(0, c.noise, true_feats.shape).to(true_feats.device)
                gaus_feats = (true_feats + noise).detach().requires_grad_(True)
                center = gnet_model.center.repeat(gaus_feats.shape[0], 1)
                for step in range(c.step + 1):
                    gaus_scores = gnet_model.discriminator(gaus_feats)
                    loss_bce_gaus = F.binary_cross_entropy(gaus_scores, torch.ones_like(gaus_scores))
                    if step == c.step: break
                    grad, = torch.autograd.grad(loss_bce_gaus, [gaus_feats], retain_graph=True)
                    with torch.no_grad():
                        grad_norm = torch.norm(grad, p=2, dim=1).view(-1, 1)
                        grad_normalized = grad / (grad_norm + 1e-10)
                        gaus_feats.add_(0.001 * grad_normalized)
                    if (step + 1) % 5 == 0:
                        with torch.no_grad():
                            dist_t = torch.norm(true_feats - center, dim=1)
                            r_t = torch.quantile(dist_t, q=c.radius)
                            h = gaus_feats - center
                            h_norm = torch.norm(h, dim=1)
                            alpha = torch.clamp(h_norm, r_t, 2 * r_t)
                            proj_factor = (alpha / (h_norm + 1e-10)).unsqueeze(1)
                            h_proj = proj_factor * h
                            gaus_feats.copy_(center + h_proj)
            else:
                gaus_feats = None

            fake_scores = gnet_model.discriminator(fake_feats)
            mask_s_gt = mask_s.reshape(-1, 1).to(fake_scores.device)

            if c.bce_loss:
                if fake_scores.numel() > 0:
                    loss_focal = F.binary_cross_entropy(fake_scores, mask_s_gt.float())
                else:
                    loss_focal = torch.tensor(0.0, device=device)
            else:
                if c.p > 0 and fake_scores.numel() > 0:
                    with torch.no_grad():
                        fake_dist = (fake_scores - mask_s_gt) ** 2
                        if fake_dist.numel() > 0:
                            d_hard = torch.quantile(fake_dist, q=c.p)
                            hard_indices = fake_dist >= d_hard
                        else:
                            hard_indices = torch.zeros_like(fake_dist, dtype=torch.bool)
                    if hard_indices.sum() > 0:
                        fake_scores_hard = fake_scores[hard_indices].unsqueeze(1)
                        mask_s_gt_hard = mask_s_gt[hard_indices].unsqueeze(1)
                        focal_logits = torch.cat([1 - fake_scores_hard, fake_scores_hard], dim=1)
                        loss_focal, f_val, d_val = gnet_model.focal_loss(focal_logits, mask_s_gt_hard.long())
                    else:
                        loss_focal = torch.tensor(0.0, device=device)
                else:
                    if fake_scores.numel() > 0 and mask_s_gt.numel() > 0:
                        focal_logits = torch.cat([1 - fake_scores, fake_scores], dim=1)
                        loss_focal, f_val, d_val = gnet_model.focal_loss(focal_logits, mask_s_gt.long())
                    else:
                        loss_focal = torch.tensor(0.0, device=device)

            true_scores = gnet_model.discriminator(true_feats)
            loss_bce_true = F.binary_cross_entropy(true_scores, torch.zeros_like(true_scores))

            if not c.no_noise and gaus_feats is not None:
                gaus_scores_final = gnet_model.discriminator(gaus_feats.detach())
                loss_bce_gaus_final = F.binary_cross_entropy(gaus_scores_final, torch.ones_like(gaus_scores_final))
                bce_loss = loss_bce_true + loss_bce_gaus_final
            else:
                bce_loss = loss_bce_true

            total_loss = bce_loss + loss_focal + 0.1 * svdd_loss

            if c.use_dfs and dfs_weights is not None:
                for w in dfs_weights:
                    total_loss -= c.entropy_weight * (w * torch.log(w + 1e-8)).sum(dim=1).mean()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gnet_model.pre_projection.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(gnet_model.discriminator.parameters(), max_norm=1.0)

            main_opt.step()
            proj_opt.step()
            dsc_opt.step()

            total_loss_epoch += total_loss.item()
            total_svdd_epoch += svdd_loss.item()

            curr_alpha = 0.0
            if hasattr(gnet_model.s, 'attention') and isinstance(gnet_model.s.attention, SelectiveHybridAttention):
                curr_alpha = gnet_model.s.attention.last_avg_alpha
                avg_alpha_epoch += curr_alpha
                batch_count += 1

            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", svdd=f"{svdd_loss.item():.4f}",
                                     alpha=f"{curr_alpha:.3f}")

        avg_loss = total_loss_epoch / len(train_dataloader)
        avg_svdd = total_svdd_epoch / len(train_dataloader)

        print(f"Epoch {epoch + 1} finished")
        print(f"    Total Loss: {avg_loss:.4f} | SVDD: {avg_svdd:.6f}")

        if batch_count > 0:
            final_alpha = avg_alpha_epoch / batch_count
            bias = "Structure (CA)" if final_alpha > 0.5 else "Texture (SE)"
            print(f"    Fusion Monitor: Mean Channel Weight: {final_alpha:.4f} -> Tendency: {bias}")

        scheduler_proj.step()
        scheduler_dsc.step()
        scheduler_main.step()

        if (epoch + 1) % c.eval_interval == 0:
            print(f"\nEvaluating at epoch {epoch + 1}")
            i_auroc, i_aupro, p_auroc, p_pro, p_ap = test_model(c, gnet_model, test_dataloader, device,
                                                                save_images=False)
            print(f"Evaluation results - Image AUROC: {i_auroc:.2f}, Image AUPRO: {i_aupro:.2f}, "
                  f"Pixel AUROC: {p_auroc:.2f}, Pixel PRO: {p_pro:.2f}, Pixel AP: {p_ap:.2f}")

            current_score = i_auroc + p_auroc
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                print(f"New best performance found at epoch {best_epoch}")
                if c.is_saved:
                    save_path = os.path.join(c.save_dir, 'checkpoints', f'gnet_{c._class_}_best.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(gnet_model.state_dict(), save_path)
                    print(f"Best model saved to: {save_path}\n")

    print(f"Training for class '{c._class_}' complete! Best model found at epoch {best_epoch}")
