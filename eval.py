import glob
import math
import os
import re
import time
import torch
import numpy as np
from skimage.measure import regionprops
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score, f1_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from sklearn import manifold, metrics
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib

from utils import t2np, rescale
import sys
from functools import partial
from multiprocessing import Pool
from skimage.measure import label, regionprops

from video_dataset import Label_loader
from NanoAD_lib.mechanism import weighted_decision_mechanism


class Config:
    dataset = "MVTec AD"
    _class_ = "bottle"
    setting = "oc"
    domain = "industrial"
    image_size = 256
    batch_size = 8
    alpha = 0.01
    beta = 0.00003


def evaluation_indusAD(c, model, test_dataloader, dataloader, device, return_heatmaps=False):
    model.train_or_eval(type='eval')
    n = model.n
    is_similarity = c.weighted_decision_mechanism
    gt_list_px = []
    gt_list_sp = []
    output_list = [list() for _ in range(n * 3)]
    weights_cnt = 0
    start_time = time.time()
    heatmap_data = []

    if return_heatmaps:
        for (x, y, mask) in test_dataloader:
            with torch.no_grad():
                t_tf, de_features = model(x.to(device))
                output = 1 - F.cosine_similarity(t_tf[0], de_features[0])
                anomaly_map = output.squeeze().cpu().numpy()

            anomaly_map = F.interpolate(
                torch.tensor(anomaly_map).unsqueeze(0).unsqueeze(0),
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            anomaly_map = anomaly_map.squeeze().numpy()
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            for i in range(x.shape[0]):
                heatmap_data.append((
                    x[i].cpu(),
                    anomaly_map[i] if len(anomaly_map.shape) > 2 else anomaly_map,
                    y[i]
                ))
    with torch.no_grad():
        for idx, (sample, label, gt) in enumerate(dataloader):

            gt_list_sp.extend(t2np(label))
            gt_list_px.extend(t2np(gt))
            weights_cnt += 1

            img = sample[0].to(device) if c.dataset == "MVTec 3D-AD" else sample.to(device)
            t_tf, de_features = model(img)

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)
                output_list[l].append(output)
        fps = len(dataloader.dataset) / (time.time() - start_time)
        print("fps:", fps, len(dataloader.dataset))

        anomaly_score, anomaly_map = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)

        gt_label = np.asarray(gt_list_sp, dtype=np.bool_)
        gt_mask_unified = []

        for gt_item in gt_list_px:
            if hasattr(gt_item, 'shape'):
                if len(gt_item.shape) == 3:
                    mask_2d = gt_item[0].astype(bool)
                elif len(gt_item.shape) == 2:
                    mask_2d = gt_item.astype(bool)
                else:
                    mask_2d = gt_item.squeeze().astype(bool)
                    if len(mask_2d.shape) > 2:
                        mask_2d = mask_2d[0] if mask_2d.shape[0] <= mask_2d.shape[-1] else mask_2d.flatten()[
                                                                                           :256 * 256].reshape(256, 256)

                gt_mask_unified.append(mask_2d)
            else:
                gt_mask_unified.append(np.zeros((256, 256), dtype=bool))

        print("Verifying shape consistency:")
        for i, mask in enumerate(gt_mask_unified[:3]):
            print(f"  mask {i}: {mask.shape}")

        gt_mask = np.array(gt_mask_unified, dtype=np.bool_)
        print(f"Final gt_mask shape: {gt_mask.shape}")

        auroc_px = round(roc_auc_score(gt_mask.flatten(), anomaly_map.flatten()) * 100, 1)
        auroc_sp = round(roc_auc_score(gt_label, anomaly_score) * 100, 1)

        pro = round(eval_seg_pro(gt_mask, anomaly_map), 1)

    if return_heatmaps:
        return auroc_px, auroc_sp, pro, heatmap_data
    else:
        return auroc_px, auroc_sp, pro


def evaluation_vad(c, model, dataloader, device):
    model.train_or_eval(type='eval')
    n = model.n
    gt_list_sp = []
    pr_list_sp = []
    output_list = [list() for _ in range(n*3)]
    weights_cnt = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(dataloader):

            img, label = img.to(device), label.to(device)
            t_tf, de_features, _ = model(img)

            label[label > 0.5] = 1
            gt_list_sp.extend(t2np(label))
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)
                output_list[l].append(output)

        if c.weighted_decision_mechanism:
            anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
            pr_list_sp = anomaly_score

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh) * 100
        f1 = f1_score(gt_list_sp, pr_list_sp >= thresh) * 100
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4) * 100

    return auroc_sp, f1, acc


def eval_seg_pro(gt_mask, anomaly_score_map, max_step=800):
    expect_fpr = 0.3

    max_th = anomaly_score_map.max()
    min_th = anomaly_score_map.min()
    delta = (max_th - min_th) / max_step
    threds = np.arange(min_th, max_th, delta).tolist()

    pool = Pool(8)
    ret = pool.map(partial(single_process, anomaly_score_map, gt_mask), threds)
    pool.close()
    pros_mean = []
    fprs = []
    for pro_mean, fpr in ret:
        pros_mean.append(pro_mean)
        fprs.append(fpr)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)
    idx = fprs < expect_fpr
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = pros_mean[idx]
    loc_pro_auc = auc(fprs_selected, pros_mean_selected) * 100

    return loc_pro_auc


def single_process(anomaly_score_map, gt_mask, thred):
    binary_score_maps = np.zeros_like(anomaly_score_map, dtype=np.bool_)
    binary_score_maps[anomaly_score_map <= thred] = 0
    binary_score_maps[anomaly_score_map > thred] = 1
    pro = []
    for binary_map, mask in zip(binary_score_maps, gt_mask):
        for region in regionprops(label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pro.append(tp_pixels / region.area)

    pros_mean = np.array(pro).mean()
    inverse_masks = 1 - gt_mask
    fpr = np.logical_and(inverse_masks, binary_score_maps).sum() / inverse_masks.sum()
    return pros_mean, fpr


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def evaluation_batch(c, model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    model.train_or_eval(type='eval')
    gt_list_sp = []
    output_list = [list() for i in range(6)]
    weights_cnt = 0
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, cls in dataloader:
            img = img.to(device)
            gt_list_sp.extend(t2np(label))
            t_tf, de_features = model(img)
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)
                output_list[l].append(output)

        anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)

        gt_list_sp = np.asarray(gt_list_sp, dtype=np.bool_)

        auroc_sp = round(roc_auc_score(gt_list_sp, anomaly_score), 4)
        ap_sp = round(average_precision_score(gt_list_sp, anomaly_score), 4)
        f1_sp = f1_score_max(gt_list_sp, anomaly_score)

    return auroc_sp, ap_sp, f1_sp


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def evaluation_mediAD(c, model, dataloader, device, _class_=None, reduction='max'):
    model.train_or_eval(type='eval')
    n = model.n
    weights_cnt = 0
    output_list = [list() for _ in range(n*3)]
    gt_list_sp = []
    pr_list_sp = []
    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            t_tf, de_features = model(img)

            gt_list_sp.extend(t2np(label))
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t_tf, de_features)):
                output = 1 - F.cosine_similarity(t, s)
                output_list[l].append(output)

        if c.weighted_decision_mechanism:
            anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
            pr_list_sp = anomaly_score

        thresh = return_best_thr(gt_list_sp, pr_list_sp)
        acc = accuracy_score(gt_list_sp, pr_list_sp >= thresh) * 100
        f1 = f1_score(gt_list_sp, pr_list_sp >= thresh) * 100
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4) * 100
    return auroc_sp, f1, acc


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def evaluation_polypseg(c, model, test_dataset, num1, trainsize=256):
    model.train_or_eval(type='eval')
    DSC = 0.0
    IOU = 0.0
    n = model.n

    for i in range(num1):
        image, gt, name = test_dataset.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        t1, de_features, recon = model(image)

        res = F.interpolate((recon[0][0]+recon[0][-1]),
                            size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

        iou = (intersection.sum() + smooth) / \
              (input.sum() + target.sum() - intersection.sum() + smooth)
        IOU = IOU + iou

    return DSC / num1, IOU / num1


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr+1e-8)))


def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))

    return anomaly_score_list


def evaluation_video(c, model, test_folder, dataloader, device):
    from collections import OrderedDict
    test_folders = os.listdir(test_folder)
    test_folders = sorted(test_folders, key=extract_numbers)
    test_folders = [os.path.join(test_folder, aa) for aa in test_folders]
    test_length = len(test_folders)
    gt_loader = Label_loader(c, test_folders)
    gt = gt_loader()
    labels_list = np.load('video/ped2/frame_labels_ped2.npy')

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    label_length = 0
    list1 = {}
    list2 = {}
    list3 = {}
    list4 = {}
    list5 = {}

    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        label_length += videos[video_name]['length']
        list1[video_name] = []
        list2[video_name] = []
        list3[video_name] = []
        list4[video_name] = []
        list5[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    model.train_or_eval(type='eval')
    n = model.n
    weights_cnt = 0
    output_list = [list() for _ in range(n * 3)]
    recon_list1 = []
    test_length_list = []
    test_length_list.append(label_length)
    with torch.no_grad():
        for k, (imgs, _) in enumerate(dataloader):
            if k == label_length:
                video_num += 1
                label_length += videos[videos_list[video_num].split('/')[-1]]['length']
                test_length_list.append(videos[videos_list[video_num].split('/')[-1]]['length'])

            imgs = (imgs).cuda()
            t1, de_features, pred = model(imgs)
            weights_cnt += 1

            for l, (t, s) in enumerate(zip(t1, de_features)):
                output = 1 - F.cosine_similarity(t, s)
                output_list[l].append(output)

            recon_list1.append((F.mse_loss(pred[0], imgs[:, -3:]) + F.mse_loss(pred[-1], imgs[:, -3:])).detach().cpu().numpy())

        anomaly_score, _ = weighted_decision_mechanism(weights_cnt, output_list, c.alpha, c.beta)
        anomaly_map = anomaly_score

        anomaly_list1 = []
        anomaly_list2 = []
        anomaly_list3 = []
        anomaly_list4 = []
        anomaly_list5 = []

        for video in sorted(videos_list):
            break

        scores = np.array([], dtype=np.float32)
        labels = np.array([], dtype=np.int8)
        start = 0
        end = test_length_list[0]
        for i in range(test_length):
            score = []
            for j in range(start, end):
                score.append(anomaly_map[j][0] * 1 + recon_list1[j] * 1)

            scores = np.concatenate((scores, score), axis=0)

            label = gt[i][:len(score)]
            labels = np.concatenate((labels, label), axis=0)

            if i+1 < test_length:
                start = start + end
                end = end + test_length_list[i+1]
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        accuracy = metrics.auc(fpr, tpr)

        return accuracy * 100
