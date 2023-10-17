from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
from tqdm import tqdm, trange
import time

import albumentations as A
import cv2

from PatchApplier import PatchApplier

import sys
sys.path.append('./yolov7')

import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


transforms = A.Compose([
        A.LongestMaxSize(640),
        A.PadIfNeeded(640,640, border_mode=cv2.BORDER_CONSTANT)
    ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1))

def do_transform(img, anno):
    img, xywh = transforms(
        image=np.array(
            img.getdata())
        .reshape(img.height, img.width, 3)
        .astype(np.uint8), 
        bboxes=[[
            max(0.0, a['bbox'][0]),
            max(0.0, a['bbox'][1]),
            min(a['bbox'][2], img.width) - a['bbox'][0],
            min(a['bbox'][3], img.height) - a['bbox'][1],
            a['category_id']
        ] for a in anno]
    ).values()
    img = torch.tensor(img).permute(2,0,1)/255
    return img, np.array([a[:-1] for a in xywh]).reshape(-1,4).astype(np.int64)
    

ds = torchvision.datasets.CocoDetection(
    '/data/INRIAPerson/Train/pos', 
    'data/inriaperson_train.json', 
    transforms=do_transform)


device = select_device('0')
#device = 'cuda:0,1,2'
imgsz=640

# Load model
model = attempt_load('./yolov7/yolov7.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def collate(data):
    #images = torch.stack([T.functional.to_tensor(img) for img, _ in data])
    images = torch.stack([img for img, _ in data])#torch.tensor(np.array([img for img, _ in data])).permute(0,3,1,2)/255
    xyxy = [xyxy for _, xyxy in data]
    return images, xyxy

dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, pin_memory=True, num_workers=16, collate_fn=collate)
dl_test = torch.utils.data.DataLoader(ds, batch_size=1)

model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
old_img_w = old_img_h = imgsz
old_img_b = 1

class Meter:
    def __init__(self, size=100):
        self.values = []
        self.size = size
        
    def __call__(self, value):
        if len(self.values) > self.size:
            self.values = self.values[1:]
        self.values.append(value)
        return self.get()
    
    def get(self):
        return np.nanmean(self.values)
    
#def train():
def smoothness(patch):
    return (patch[:,:,1:] - patch[:,:,:-1]).abs().mean() + (patch[:,1:] - patch[:,:-1]).abs().mean()


def objectness(pred, τ_obj=0.75, τ_cls=0.75, target_cls=0):
    objectness, cls = pred[...,4], pred[...,5:]
    target_filter = pred[...,5:].argmax(2) == target_cls
    
    valid = torch.logical_and(objectness.sigmoid() > τ_obj, cls[..., target_cls].sigmoid() > τ_cls)
    valid = torch.logical_and(target_filter, valid)
    return objectness[valid].max() if objectness[valid].numel() > 0 else objectness.max()# * cls[..., target_cls][valid]).mean()
    
def validity(patch):
    return torch.nn.functional.mse_loss(patch, patch.clamp(0.05,0.95))

normalize = lambda x : (x-x.min()) / (x.max() - x.min())

cudnn.benchmark = True  # set True to speed up constant image size inference
@torch.cuda.amp.autocast()
def train(epoch):
    meters = {
        key : Meter() for key in ['obj', 'smt', 'val']
    }
    for i, (imgs, labels) in enumerate(dl):
        optimizer.zero_grad()
        
        imgs = torch.stack([applier(img, torchvision.ops.box_convert(torch.from_numpy(label), in_fmt='xywh', out_fmt='xyxy'), patch=patch, normalized_annotations=False) for img, label in zip(imgs, labels)])
        pred = model(imgs.to(device), augment=True)[0]

        obj = objectness(pred, τ_obj=0, τ_cls=0)
        smt = smoothness(patch)
        val = 2 * validity(patch)

        loss = obj + smt + val
        
        loss.backward()
        optimizer.step()
        meters['obj'](obj.item())
        meters['smt'](smt.item())
        meters['val'](val.item())

    return {key: meters[key].get() for key in meters}


from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix

### YOLOv7 evaluation code

config = {
    'width': 256,
    'height': 256,
    'epochs': 1,
    'num_patches': 3,
    'timestamp': time.time(),
    
    'optimizer': {
        'type': 'AdamW',
        'parameters': {
            'lr': 0.01,
        }
    },
    'scheduler': {
        'type': 'StepLR',
        'parameters': {
            'step_size': 25,
        }
    },
    'applier': {
        'mode': 'MODE_RANDOM_BBOX',
        'resize_range': [0.5, 0.75],
        'transforms': [
            {
                'type': 'ColorJitter',
                'parameters': {
                    'brightness': 0.1,
                    'contrast': 0.05,
                    'saturation': 0.03,
                    'hue': 0,
                }
            },
            {
                'type': 'RandomRotation',
                'parameters': {
                    'degrees': 45,
                }
            },
            {
                'type': 'RandomPerspective',
                'parameters': {
                    'distortion_scale': 0.5
                }
            }
        ]
    },
}

output_path = Path('outputs')
output_path.mkdir(exist_ok=True)
patches = []
losses = []
for i in range(config['num_patches']):
    patch = torch.rand([3, config['height'], config['width'] ]).requires_grad_()
    optimizer = getattr(torch.optim, config['optimizer']['type'])([patch], **(config['optimizer']['parameters']))
    scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['type'])(optimizer, **(config['scheduler']['parameters']))

    applier = PatchApplier(
        patch, 
        patch_transforms=T.Compose([
            getattr(T, transform['type'])(**(transform['parameters']))
            for transform in config['applier']['transforms']
        ]),
        resize_range=config['applier']['resize_range'],
        mode=getattr(PatchApplier, config['applier']['mode'])
    )
    
    tr = trange(config['epochs'])
    for epoch in tr:
        loss = train(epoch)
        scheduler.step()
        T.ToPILImage()(torchvision.utils.make_grid(patch.unsqueeze(0), nrow=4).detach().cpu()).save(f'./patch_{config["timestamp"]}.png')
        tr.set_postfix(loss)

    patches.append(patch.detach().cpu())
    losses.append(loss)
    
torch.save({
    'config': config,
    'patches': torch.stack(patches),
    'losses': losses,
}, output_path / f'{config["timestamp"]}_patches.pth')
