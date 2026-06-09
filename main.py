import numpy as np
import os
import argparse
import pandas as pd
from tabulate import tabulate
import torch
from torch import nn
from torchvision import models

from datasets import mvtec_list, btad_list, visa_list, mpdd_list, mvtec_loco_list, get_dataloaders_for_gnet
from utils import setup_seed
from train_NanoAD import train
from NanoAD import NanoAD
from test import test_model


def parsing_args():
    parser = argparse.ArgumentParser(description='GNet Training and Testing')
    parser.add_argument('--dataset', default='MVTec AD', type=str,
                        choices=['MVTec AD', 'BTAD', 'VisA', 'MPDD', 'MVTec LOCO'])
    parser.add_argument('--data_path', type=str, required=True, help='Root path of the dataset.')
    parser.add_argument('--anomaly_source_path', type=str, help='Path to DTD texture images (required for training).')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save results. If None, generated based on dataset.')
    parser.add_argument('--load_ckpts', action='store_true', default=False, help="Set to true to only run testing.")
    parser.add_argument('--is_saved', action='store_true', default=True)
    parser.add_argument('--class_name', type=str, default=None, help="Specify a single class to train/test.")

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--backbone', default='wideresnet50', type=str)
    parser.add_argument('--layers_to_extract', nargs='+', default=['layer2', 'layer3'])
    parser.add_argument('--pretrain_embed_dimension', default=1536, type=int)
    parser.add_argument('--target_embed_dimension', default=1536, type=int)
    parser.add_argument('--patchsize', default=3, type=int)
    parser.add_argument('--pre_proj', default=1, type=int)

    parser.add_argument('--dsc_layers', default=2, type=int)
    parser.add_argument('--dsc_hidden', default=1024, type=int)
    parser.add_argument('--noise', default=0.015, type=float)
    parser.add_argument('--step', default=20, type=int)
    parser.add_argument('--radius', type=float, default=0.75)

    parser.add_argument('--use_dfs', action='store_true', default=False)
    parser.add_argument('--dfs_learnable', action='store_true', default=False)
    parser.add_argument('--entropy_weight', type=float, default=0.001,
                        help="Weight for the DFS weights entropy regularization loss.")

    parser.add_argument('--ablation_no_bn_student', action='store_true', default=False,
                        help="Ablation study: Replace BN and Adapter modules with Identity.")

    parser.add_argument('--image_size', default=288, type=int)
    parser.add_argument('--p', type=float, default=0.5)

    parser.add_argument('--no_residual', action='store_true', default=False,
                        help="Ablation: Disable Learnable Residual Module (use raw features instead).")

    parser.add_argument('--no_adapter', action='store_true', default=False,
                        help="Ablation: Disable Adapter Module (use Identity instead).")

    parser.add_argument('--no_kmeans', action='store_true', default=False,
                        help="Ablation: Disable K-Means initialization for prototypes (use Random).")

    parser.add_argument('--svdd_type', type=str, default='residual',
                        choices=['residual', 'traditional', 'multi_center'],
                        help="Ablation: Choose 'residual' (LRPM), 'multi_center' (multi-center without residual mapping) or 'traditional' (single center).")

    parser.add_argument('--no_pe', action='store_true', default=False,
                        help="Ablation: Disable Positional Encoding in Adapter.")

    parser.add_argument('--no_dtd', action='store_true', default=False,
                        help="Ablation: Disable DTD synthetic anomalies (use normal images instead).")

    parser.add_argument('--no_noise', action='store_true', default=False,
                        help="Ablation: Disable feature noise/perturbation in training.")

    parser.add_argument('--bce_loss', action='store_true', default=False,
                        help="Ablation: Use standard BCE Loss instead of Focal+Dice for DTD anomalies.")

    args = parser.parse_args()
    return args


def print_model_parameters(model, model_name="NanoAD"):
    print(f"\n{'=' * 20} {model_name} Model Parameters {'=' * 20}")

    total_params = 0
    total_trainable_params = 0
    table_data = []

    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters())
        child_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)

        table_data.append([
            name,
            f"{child_params:,}",
            f"{child_trainable:,}",
            f"{child_params / 1e6:.2f} M"
        ])

        total_params += child_params
        total_trainable_params += child_trainable

    print(tabulate(table_data, headers=["Module", "Total Params", "Trainable Params", "Params (M)"], tablefmt="simple"))
    print("-" * 65)

    print(f"Total Parameters:     {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable Parameters: {total_trainable_params:,} ({total_trainable_params / 1e6:.2f} M)")

    size_mb = total_params * 4 / (1024 ** 2)
    print(f"Estimated Size:       {size_mb:.2f} MB")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    setup_seed(1203)
    c = parsing_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if c.save_dir is None:
        dataset_name_clean = c.dataset.replace(' ', '_')
        c.save_dir = f'./results/GNet_{dataset_name_clean}'

    print(f"Training results and model will save to: {c.save_dir}")
    os.makedirs(c.save_dir, exist_ok=True)

    if c.use_dfs:
        if c.backbone == 'wideresnet50':
            channel_map = {'layer1': 256, 'layer2': 512, 'layer3': 1024}
        elif c.backbone == 'resnet18':
            channel_map = {'layer1': 64, 'layer2': 128, 'layer3': 256}
        else:
            raise ValueError(f"Backbone {c.backbone} not supported for DFS channel mapping.")
        c.dfs_channels = [channel_map[layer] for layer in c.layers_to_extract]
        print(f"DFS enabled with channels: {c.dfs_channels} for backbone {c.backbone}")

    if c.class_name:
        dataset_classes = [c.class_name]
    else:
        if c.dataset == 'MVTec AD':
            dataset_classes = mvtec_list
        elif c.dataset == 'BTAD':
            dataset_classes = btad_list
        elif c.dataset == 'VisA':
            dataset_classes = visa_list
        elif c.dataset == 'MPDD':
            dataset_classes = mpdd_list
        elif c.dataset == 'MVTec LOCO':
            dataset_classes = mvtec_loco_list
        else:
            raise ValueError(f"Unknown dataset: {c.dataset}")

    if not c.load_ckpts:
        print("Starting Training Phase")
        for class_name in dataset_classes:
            c._class_ = class_name
            print(f"\nTraining class: {class_name}")
            train(c)
        print("\nAll Training Finished")

    print("\nStarting Final Testing Phase")
    results_summary = []
    headers = ['object', 'image_auroc', 'image_aupro', 'pixel_auroc', 'pixel_aupro', 'pixel_ap']
    image_auroc_list, image_aupro_list, pixel_auroc_list, pixel_aupro_list, pixel_ap_list = [], [], [], [], []

    for class_name in dataset_classes:
        c._class_ = class_name
        print(f"\nTesting class: {class_name}")

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
        model = NanoAD(c=c, Feature_Aggregator_params=Feature_Aggregator_params, bottleneck=bottleneck_model,
                       student=student_model).to(device)

        ckpt_path = os.path.join(c.save_dir, 'checkpoints', f'gnet_{c._class_}_best.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found for class '{class_name}' at: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Model loaded from {ckpt_path}")

        print_model_parameters(model)

        _, test_dataloader = get_dataloaders_for_gnet(c)
        i_auroc, i_aupro, p_auroc, p_pro, p_ap = test_model(c, model, test_dataloader, device, save_images=True)

        results_summary.append(
            [class_name, f"{i_auroc:.2f}", f"{i_aupro:.2f}", f"{p_auroc:.2f}", f"{p_pro:.2f}", f"{p_ap:.2f}"])
        image_auroc_list.append(i_auroc)
        image_aupro_list.append(i_aupro)
        pixel_auroc_list.append(p_auroc)
        pixel_aupro_list.append(p_pro)
        pixel_ap_list.append(p_ap)

    if results_summary and len(results_summary) > 1:
        mean_i_auroc = np.mean(image_auroc_list)
        mean_i_aupro = np.mean(image_aupro_list)
        mean_p_auroc = np.mean(pixel_auroc_list)
        mean_p_aupro = np.mean(pixel_aupro_list)
        mean_p_ap = np.mean(pixel_ap_list)
        results_summary.append(
            ['mean', f"{mean_i_auroc:.2f}", f"{mean_i_aupro:.2f}", f"{mean_p_auroc:.2f}", f"{mean_p_aupro:.2f}",
             f"{mean_p_ap:.2f}"])

    print("\n\nFinal Results Summary")
    print(tabulate(results_summary, headers=headers, tablefmt="pipe"))
