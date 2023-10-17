import numpy as np
from sklearn.decomposition import PCA

class PatchSampler:
    def __init__(self, patches, n_components=16):
        self.n_components = n_components
        self.shape = patches.shape[1:]
        self.transformed = None
        self.means, self.scale = None, None
        self.fit(patches)

    def fit(self, patches):
        self.shape = patches.shape[1:]
        self.pca = PCA(self.n_components)
        self.transformed = self.pca.fit_transform(patches.flatten(1))
        self.means, self.scale = self.transformed.mean(0), self.transformed.std(0)
        return self

    def sample(self):
        sample = np.random.normal(self.means, self.scale)
        patches = self.pca.inverse_transform(sample).reshape(*self.shape).clip(0,1)
        return patches

if __name__ == '__main__':
    import argparse
    from PIL import Image
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='patches.pth')
    parser.add_argument('-c', '--components', default=16)
    parser.add_argument('-o', '--output', default='sampled_patch.png')
    args = parser.parse_args()
    
    sampler = PatchSampler(torch.load(args.input)['patches'], args.components)
    Image.fromarray(
        (sampler.sample().transpose(1,2,0)*255).astype(np.uint8)
    ).save(args.output)
    