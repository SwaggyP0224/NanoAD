import PIL
from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from torch.utils.data import DataLoader
from enum import Enum
from perlin import perlin_mask
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

mvtec_list = ['transistor','cable', 'carpet', 'bottle', 'hazelnut', 'leather', 'capsule', 'grid', 'pill',
               'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
btad_list = ["01", "02", "03"]
visa_list = ['pipe_fryum','candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', ]
mpdd_list = ['bracket_black', 'bracket_brown', 'bracket_white',
             'connector', 'metal_plate', 'tubes']


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class GlassMVTecDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
            classname,
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            anomaly_source_path=None,
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname
        self.imagesize = imagesize
        self.resize = resize

        self.dataset_name = kwargs.get('dataset', 'MVTec AD')

        self.rand_aug = kwargs.get('rand_aug', False)
        self.class_fg = kwargs.get('class_fg', False)
        self.downsampling = kwargs.get('downsampling', 8)
        self.mean = kwargs.get('mean', 0.5)
        self.std = kwargs.get('std', 0.1)

        self.model_downsampling_factor = 4
        self.patch_grid_size = self.imagesize // self.model_downsampling_factor

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        if self.split == DatasetSplit.TRAIN:
            if anomaly_source_path is None or not os.path.exists(anomaly_source_path):
                raise ValueError(f"Anomaly source path '{anomaly_source_path}' is required.")

            dtd_paths = sorted(glob.glob(os.path.join(anomaly_source_path, "*/*.jpg")))
            if not dtd_paths:
                dtd_paths = sorted(glob.glob(os.path.join(anomaly_source_path, "*.jpg")))

            if dtd_paths:
                self.anomaly_source_paths = dtd_paths
            else:
                if 'good' in self.imgpaths_per_class[self.classname]:
                    self.anomaly_source_paths = self.imgpaths_per_class[self.classname]['good']
                else:
                    raise FileNotFoundError(f"Cannot load anomaly source and no 'good' images found.")

        self.transform_img = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        aug_image = torch.zeros_like(image)
        mask_s = torch.zeros((1, self.patch_grid_size, self.patch_grid_size), dtype=torch.float32)

        if self.split == DatasetSplit.TRAIN:
            aug_path = np.random.choice(self.anomaly_source_paths)
            aug = PIL.Image.open(aug_path).convert("RGB")
            aug = self.transform_img(aug)

            mask_all = perlin_mask(image.shape, self.imagesize // self.downsampling, 0, 6, torch.ones_like(image[0]), 1)
            mask_s = torch.from_numpy(mask_all[0]).float().unsqueeze(0)
            mask_l = torch.from_numpy(mask_all[1]).float()

            beta = np.float32(np.random.normal(loc=self.mean, scale=self.std))
            beta = np.clip(beta, .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        mask_gt = torch.zeros([1, *image.size()[1:]])
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt_pil = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt_pil)

            mask_gt[mask_gt > 0.0001] = 1.0

        anomaly_lower = anomaly.lower()
        is_normal = (anomaly_lower == "good") or (anomaly_lower == "ok") or (anomaly_lower == "normal")

        return {
            "image": image, "aug": aug_image, "mask_s": mask_s, "mask_gt": mask_gt,
            "is_anomaly": int(not is_normal), "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        if self.dataset_name == 'VisA':
            return self.get_visa_data()

        imgpaths_per_class = {}
        maskpaths_per_class = {}
        classpath = os.path.join(self.source, self.classname, self.split.value)
        maskpath = os.path.join(self.source, self.classname, "ground_truth")

        anomaly_types = os.listdir(classpath)
        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}
        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            if not os.path.isdir(anomaly_path): continue
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

            if self.split == DatasetSplit.TEST and anomaly != "good":
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                if os.path.exists(anomaly_mask_path):
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in
                                                                    anomaly_mask_files]
                else:
                    maskpaths_per_class[self.classname][anomaly] = [None] * len(anomaly_files)
            else:
                maskpaths_per_class[self.classname][anomaly] = [None] * len(anomaly_files)

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[self.classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)
        return imgpaths_per_class, data_to_iterate

    def get_visa_data(self):
        imgpaths_per_class = {self.classname: {}}
        maskpaths_per_class = {self.classname: {}}
        data_to_iterate = []

        base_path = os.path.join(self.source, self.classname, 'Data')
        img_dir = os.path.join(base_path, 'Images')
        mask_dir = os.path.join(base_path, 'Masks', 'Anomaly')

        normal_imgs = sorted(glob.glob(os.path.join(img_dir, 'Normal', '*')))
        anomaly_imgs = sorted(glob.glob(os.path.join(img_dir, 'Anomaly', '*')))

        split_ratio = 0.8
        split_idx = int(len(normal_imgs) * split_ratio)

        if self.split == DatasetSplit.TRAIN:
            imgpaths_per_class[self.classname]['good'] = normal_imgs[:split_idx]
            maskpaths_per_class[self.classname]['good'] = [None] * len(normal_imgs[:split_idx])
        else:
            imgpaths_per_class[self.classname]['good'] = normal_imgs[split_idx:]
            maskpaths_per_class[self.classname]['good'] = [None] * len(normal_imgs[split_idx:])

            imgpaths_per_class[self.classname]['Anomaly'] = anomaly_imgs
            masks = []
            for img_path in anomaly_imgs:
                file_name = os.path.basename(img_path)
                file_name_no_ext = os.path.splitext(file_name)[0]
                mask_path = os.path.join(mask_dir, file_name_no_ext + '.png')

                if os.path.exists(mask_path):
                    masks.append(mask_path)
                else:
                    mask_path_jpg = os.path.join(mask_dir, file_name_no_ext + '.JPG')
                    if os.path.exists(mask_path_jpg):
                        masks.append(mask_path_jpg)
                    else:
                        print(f"Warning: Mask not found for {img_path}")
                        masks.append(None)
            maskpaths_per_class[self.classname]['Anomaly'] = masks

        for anomaly_type in imgpaths_per_class[self.classname].keys():
            paths = imgpaths_per_class[self.classname][anomaly_type]
            masks = maskpaths_per_class[self.classname][anomaly_type]

            for i, img_path in enumerate(paths):
                data_tuple = [self.classname, anomaly_type, img_path, masks[i]]
                data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


def get_dataloaders_for_gnet(c):
    data_source_path = getattr(c, 'data_path', './mvtec')
    base_args = vars(c).copy()

    base_args.update({
        'source': data_source_path,
        'classname': c._class_,
        'resize': c.image_size,
        'imagesize': c.image_size,
        'dataset': getattr(c, 'dataset', 'MVTec AD')
    })

    train_dataloader = None
    if not getattr(c, 'load_ckpts', False):
        train_args = base_args.copy()
        train_args['split'] = DatasetSplit.TRAIN
        train_dataset = GlassMVTecDataset(**train_args)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True, drop_last=True)

    test_args = base_args.copy()
    test_args['split'] = DatasetSplit.TEST
    test_dataset = GlassMVTecDataset(**test_args)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True, drop_last=False)
    return train_dataloader, test_dataloader
