import numpy as np
import torch

class PatchApplier(torch.nn.Module):
    MODE_CENTER_BBOX = 0
    MODE_RANDOM_BBOX = 1
    MODE_RANDOM = 2
    MODE_FIXED_BOTTOM = 3

    def __init__(self, patch=None, patch_transforms=None, mode=0, resize_range=(0.25, 0.75)):
        super().__init__()
        self.patch = patch
        self.patch_transforms = patch_transforms
        self.mode = mode
        self.resize_range = resize_range


    def mode_center_bbox(self, xywh, wh):
        x = (xywh[0] + xywh[2]//2) - wh//2
        y = (xywh[1] + xywh[3]//2) - wh//2

        return x, y


    def mode_random_bbox(self, xywh, wh):
        x, y = torch.rand([2])
        x = (xywh[0] + (xywh[2] - wh) * x).to(torch.int64)
        y = (xywh[1] + (xywh[3] - wh) * y).to(torch.int64)
        return x, y


    def mode_random(self, img_wh, wh):
        return self.mode_random_bbox([0,0,*img_wh], wh)


    def random_wh(self, xywh, min_size=10):
        wh = min(xywh[2], xywh[3])
        wh = wh.cuda() * (self.resize_range[0] + (self.resize_range[1] - self.resize_range[0]) * torch.rand([1]).cuda())
        wh = wh.to(torch.int64)
        wh = max(wh, min_size)
        return wh


    def xyxy_to_xywh(self, xyxy):
        return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]


    def embed_patch(self, image, patch, x, y, wh):
        device = patch.device
        y = max(min(image.shape[1] - wh, y), 0)
        x = max(min(image.shape[2] - wh, x), 0)
        
        patch_mask = torch.zeros_like(image, device=device)
        patch_mask[:, y:y+wh, x:x+wh] = 1
        image_mask = torch.ones_like(image, device=device) - patch_mask
        patch_mask[:, y:y+wh, x:x+wh] *= patch
        
        image = image_mask * image + patch_mask
        return image



    def forward(self, image, annotations, patch=None, normalized_annotations=True):
        img = image.clone()
        if normalized_annotations:
            annotations = torch.multiply(annotations, [*(image.shape[1:][::-1]), *(image.shape[1:][::-1])]).to(torch.long)
        
        if self.mode == PatchApplier.MODE_FIXED_BOTTOM:
            if patch is None:
                patch = self.patch
            if self.patch_transforms and self.training:
                patch = self.patch_transforms(self.patch)
            
            wh = torch.tensor([image.shape[1:][::-1]])
            xywh = torch.cat([wh*0.875, wh]).flatten()
            wh = self.random_wh(xywh)
            
            patch = torch.nn.functional.interpolate(patch.unsqueeze(0), (wh,wh), mode='bilinear').squeeze()
            x, y = self.mode_random_bbox(xywh, wh)
            img = self.embed_patch(img, patch, x, y, wh)
            
        else:
            patch_ = patch if patch is not None else self.patch
            for xyxy in annotations:
                patch = patch_.clone()
                if self.patch_transforms and self.training:
                    patch = self.patch_transforms(patch)

                xywh = self.xyxy_to_xywh(xyxy)
                wh = self.random_wh(xywh)
                patch = torch.nn.functional.interpolate(patch.unsqueeze(0), (wh,wh), mode='bilinear').squeeze()

                if self.mode == PatchApplier.MODE_CENTER_BBOX:
                    x, y = self.mode_center_bbox(xywh, wh)
                elif self.mode == PatchApplier.MODE_RANDOM_BBOX:
                    x, y = self.mode_random_bbox(xywh, wh)
                elif self.mode == PatchApplier.MODE_RANDOM:
                    x, y = self.mode_random(img.shape[1:][::-1], wh)

                img = self.embed_patch(img, patch, x, y, wh)
            
        return img