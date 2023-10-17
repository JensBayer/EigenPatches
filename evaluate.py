#!/usr/bin/env python
# coding: utf-8

# In[ ]:

EVALUATION_TYPE = ['nelems', 'ndims'][0]  # either nelems or ndims


from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import pandas as pd
import IPython
from tqdm import tqdm, trange
import time

import albumentations as A
import cv2

from PatchApplier import PatchApplier
import matplotlib.pyplot as plt


# In[ ]:


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
    '/data/INRIAPerson/Test/pos', 
    'data/inriaperson_test.json', 
    transforms=do_transform)


# In[ ]:


import sys
sys.path.append('./yolov7')


# In[ ]:


import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# In[ ]:


device = select_device('0')
imgsz=640

model = attempt_load('./yolov7/yolov7.pt', map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(imgsz, s=stride)

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


# In[ ]:


def collate(data):
    images = torch.stack([img for img, _ in data])
    xyxy = [xyxy for _, xyxy in data]
    return images, xyxy

dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=16, collate_fn=collate)


# In[ ]:


model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
old_img_w = old_img_h = imgsz
old_img_b = 1


# In[ ]:


normalize = lambda x : (x-x.min()) / (x.max() - x.min())


# In[ ]:


from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix


# In[ ]:


#  yolo evaluation code -- shamelessly copied <3

def test(patch_applier=None):
    seen = 0
    conf_thres = 0.5
    iou_thres = 0.5
    nc = 1

    iouv = torch.linspace(0.5, 0.95, 10).to('cuda')  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    #for batch_i, (img, targets) in enumerate(tqdm(dl, desc=s)):
    for batch_i, (img, targets) in enumerate(dl):
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        if patch_applier:
            img = torch.stack([patch_applier(img, torchvision.ops.boxes.box_convert(torch.tensor(label), 'xywh', 'xyxy'), normalized_annotations=False) for img, label in zip(img, targets)])

        img = img.cuda()
        with torch.no_grad():
            # Run model
            out, train_out = model(img, augment=True)  # inference and training outputs

            # Run NMS
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=False)


        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets #targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = [0 for _ in range(nl)]#labels[:, 0].tolist() if nl else []  # target class

            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            #scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device='cuda')
            if nl:
                detected = []  # target indices
                tcls_tensor = torch.zeros([len(labels)])#labels[:, 0]

                # target boxes
                #tbox = xywh2xyxy(labels[0])
                tbox = torchvision.ops.box_convert(torch.tensor(labels[0]), 'xywh', 'xyxy').cuda()
                #scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1).cuda()  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1).cuda()  # target indices

                    # Search for detections
                    if pi.shape[0] and tbox.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti] if len(tbox) > 0 else torch.zeros([0,4], device='cuda')).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        #break


    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, v5_metric=False, save_dir=None, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    #print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return ((mp, mr, map50, map, *(loss.cpu() / len(dl)).tolist()), maps)


# In[ ]:


output = test()
print('== baseline ==')
print(output[0][2:4])


# In[ ]:


patches = []
losses = []
patch_src = []
configs = []
for i, path in enumerate(Path('outputs').rglob('*.pth')):
    dump = torch.load(str(path))
    patches.append(dump['patches'])
    configs.append(dump['config'])
    losses += dump['losses']
    patch_src += [i for _ in range(len(dump['patches']))]
    if len(patches[-1]) != configs[-1]['num_patches']:
        print(len(patches[-1]), configs[-1]['num_patches'], path)
    elif patches[-1].isnan().any():
        print(f'{path} contains {np.sum([p.isnan().any() for p in patches[-1]])} patches with NaN. Replacing values with zero.')
        patches[-1][patches[-1].isnan()] = 0
patches = torch.cat(patches)


output = []
for patch in tqdm(patches):
    patch_applier = PatchApplier(
        patch,
        resize_range=(0.75, 0.75),
        mode=PatchApplier.MODE_CENTER_BBOX
    )
    output.append(test(patch_applier))


# In[ ]:


torch.save([
    {
        'hash': str(hash(patch)),
        'P': float(o[0][0]),
        'R': float(o[0][1]),
        'mAP@.5': float(o[0][2]),
        'mAP@.5:.95:': float(o[0][3]),
    }
    for o, patch in zip(output, patches)
], './trained.pth')


# In[ ]:


df = pd.DataFrame(torch.load('./trained.pth'))
print('== trained stats ==')
print(df.describe())


if EVALUATION_TYPE == 'ndims':

    # In[ ]:


    from sklearn.decomposition import PCA

    for pca_dim in [2, 4, 8, 16, 32, 64, 128, 256]:
        pca = PCA(pca_dim)
        reduced = pca.fit_transform(patches.flatten(1))
        recovered = pca.inverse_transform(reduced).reshape(patches.shape)
        components = pca.components_.reshape([-1,3,256,256])

        recovered_output = []
        for c in tqdm(recovered):
            patch_applier = PatchApplier(
                torch.tensor(c, dtype=torch.float),
                resize_range=(0.75, 0.75),
                mode=PatchApplier.MODE_CENTER_BBOX
            )
            recovered_output.append(test(patch_applier))

        torch.save([
            {
                'hash': str(hash(patch)),
                'P': float(o[0][0]),
                'R': float(o[0][1]),
                'mAP@.5': float(o[0][2]),
                'mAP@.5:.95:': float(o[0][3]),
            }
            for o, patch in zip(recovered_output, patches)
        ], f'./recovered_patch_stats_ndims_{pca_dim:03d}.pth')


    # In[ ]:


    df = pd.DataFrame([
        {
            'hash': str(hash(patch)),
            'P': float(o[0][0]),
            'R': float(o[0][1]),
            'mAP@.5': float(o[0][2]),
            'mAP@.5:.95:': float(o[0][3]),
        }
        for o, patch in zip(recovered_output, patches)
    ])
    print('== ndims stats ==')
    print(df.describe())

elif EVALUATION_TYPE == 'nelems':

    # In[ ]:


    pca_dim = 64


    # In[ ]:


    from sklearn.decomposition import PCA

    for patch_elems in [2, 4, 8, 16, 32, 64, 128, 256]:
        ppatches = patches[torch.randperm(len(patches))][:patch_elems]    
        pca = PCA(min(patch_elems, pca_dim))
        reduced = pca.fit(ppatches.flatten(1))
        reduced = pca.transform(patches.flatten(1))
        recovered = pca.inverse_transform(reduced).reshape(patches.shape)
        components = pca.components_.reshape([-1,3,256,256])

        recovered_output = []
        for c in tqdm(recovered):
            patch_applier = PatchApplier(
                torch.tensor(c, dtype=torch.float),
                resize_range=(0.75, 0.75),
                mode=PatchApplier.MODE_CENTER_BBOX
            )
            recovered_output.append(test(patch_applier))

    torch.save([
        {
            'hash': str(hash(patch)),
            'P': float(o[0][0]),
            'R': float(o[0][1]),
            'mAP@.5': float(o[0][2]),
            'mAP@.5:.95:': float(o[0][3]),
        }
        for o, patch in zip(recovered_output, patches)
    ], f'./recovered_patch_stats_nelems_{patch_elems:03d}.pth')


    # In[ ]:


    df = pd.DataFrame([
        {
            'hash': str(hash(patch)),
            'P': float(o[0][0]),
            'R': float(o[0][1]),
            'mAP@.5': float(o[0][2]),
            'mAP@.5:.95:': float(o[0][3]),
        }
        for o, patch in zip(recovered_output, patches)
    ])
    print('== nelems stats ==')
    print(df.describe())

