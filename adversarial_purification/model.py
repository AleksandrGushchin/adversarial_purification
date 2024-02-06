from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from defenses.conv_filter.defense import FCNDefense
from defenses.transforms.jpeg import JpegDefense
from defenses.transforms.flip import FlipDefense
from defenses.transforms.diffpure import DiffPureDefense
from defenses.transforms.upscale import UpscaleDefense
from defenses.transforms.median_filter import MedianFilterDefense
from defenses.transforms.gaussian_blur import GaussianBlurDefense
from defenses.transforms.random_crop import RandomCropDefense
from defenses.transforms.rotate import RotateDefense
from defenses.transforms.realesrgan import RealESRGANDefense
from defenses.transforms.mprnet import MPRNETDefense

from defenses.transforms.diffpure_t import DiffPureDefenseT


class ResizeDefense:
    def __init__(self):
        pass

    def __call__(self, image):
        new_size = (np.array(image.shape[2:]) * 0.5).astype(int)
        resized_image = transforms.Resize(list(new_size))(image)
        return resized_image


def SPSP(x, P=1, method='avg'):
    batch_size = x.size(0)
    map_size = x.size()[-2:]
    pool_features = []
    for p in range(1, P + 1):
        pool_size = [int(d / p) for d in map_size]
        if method == 'maxmin':
            M = F.max_pool2d(x, pool_size)
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(torch.cat((M, m), 1).view(batch_size, -1))  # max & min pooling
        elif method == 'max':
            M = F.max_pool2d(x, pool_size)
            pool_features.append(M.view(batch_size, -1))  # max pooling
        elif method == 'min':
            m = -F.max_pool2d(-x, pool_size)
            pool_features.append(m.view(batch_size, -1))  # min pooling
        elif method == 'avg':
            a = F.avg_pool2d(x, pool_size)
            pool_features.append(a.view(batch_size, -1))  # average pooling
        else:
            m1 = F.avg_pool2d(x, pool_size)
            rm2 = torch.sqrt(F.relu(F.avg_pool2d(torch.pow(x, 2), pool_size) - torch.pow(m1, 2)))
            if method == 'std':
                pool_features.append(rm2.view(batch_size, -1))  # std pooling
            else:
                pool_features.append(
                    torch.cat((m1, rm2), 1).view(batch_size, -1)
                )  # statistical pooling: mean & std
    return torch.cat(pool_features, dim=1)


class IQAModel(nn.Module):
    def __init__(
        self,
        arch='resnext101_32x8d',
        features_weights_path=None,
        pool='avg',
        use_bn_end=False,
        P6=1,
        P7=1,
    ):
        super(IQAModel, self).__init__()
        self.pool = pool
        self.use_bn_end = use_bn_end
        if pool in ['max', 'min', 'avg', 'std']:
            c = 1
        else:
            c = 2
        self.P6 = P6  #
        self.P7 = P7  #
        if features_weights_path:
            backbone = models.__dict__[arch]()
            backbone.load_state_dict(torch.load(features_weights_path))
        else:
            backbone = models.__dict__[arch](pretrained=False)
        features = list(backbone.children())[:-2]
        if arch == 'alexnet':
            in_features = [256, 256]
            self.id1 = 9
            self.id2 = 12
            features = features[0]
        elif arch == 'vgg16':
            in_features = [512, 512]
            self.id1 = 23
            self.id2 = 30
            features = features[0]
        elif 'res' in arch:
            self.id1 = 6
            self.id2 = 7
            if arch == 'resnet18' or arch == 'resnet34':
                in_features = [256, 512]
            else:
                in_features = [1024, 2048]
        else:
            print('The arch is not implemented!')
        self.features = nn.Sequential(*features)
        self.dr6 = nn.Sequential(
            nn.Linear(in_features[0] * c * sum([p * p for p in range(1, self.P6 + 1)]), 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.dr7 = nn.Sequential(
            nn.Linear(in_features[1] * c * sum([p * p for p in range(1, self.P7 + 1)]), 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        if self.use_bn_end:
            self.regr6 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regr7 = nn.Sequential(nn.Linear(64, 1), nn.BatchNorm1d(1))
            self.regression = nn.Sequential(nn.Linear(64 * 2, 1), nn.BatchNorm1d(1))
        else:
            self.regr6 = nn.Linear(64, 1)
            self.regr7 = nn.Linear(64, 1)
            self.regression = nn.Linear(64 * 2, 1)

    def extract_features(self, x):
        f, pq = [], []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.id1:
                x6 = SPSP(x, P=self.P6, method=self.pool)
                x6 = self.dr6(x6)
                f.append(x6)
                pq.append(self.regr6(x6))
            if ii == self.id2:
                x7 = SPSP(x, P=self.P7, method=self.pool)
                x7 = self.dr7(x7)
                f.append(x7)
                pq.append(self.regr7(x7))

        f = torch.cat(f, dim=1)

        return f, pq

    def forward(self, x):
        f, pq = self.extract_features(x)
        s = self.regression(f)
        pq.append(s)

        return pq


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[
            None, :, None, None
        ]


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path, defense_type=None, defense_params={'q': 50}):
        super().__init__()
        self.device = device

        model = IQAModel(arch='resnext101_32x8d')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        self.k = checkpoint['k'][0]
        self.b = checkpoint['b'][0]
        self.model = model.to(device)
        self.lower_better = False
        self.defense_type = defense_type

        if defense_type == 'baseline':
            self.defense = ResizeDefense()
        elif defense_type == 'fcn_filter':
            self.defense = FCNDefense(Path(model_path).parent / 'fcn_mse.pth', self.device)
        elif defense_type == 'jpeg':
            self.defense = JpegDefense(**defense_params)
        elif defense_type == 'flip':
            self.defense = FlipDefense(axes=[2, 3])
        elif defense_type == 'median_filter':
            self.defense = MedianFilterDefense(blur_limit=3)
        elif defense_type == 'gaussian_blur':
            self.defense = GaussianBlurDefense(blur_limit=3)
        elif defense_type == 'random_crop':
            self.defense = RandomCropDefense(size=256)
        elif defense_type == 'rotate':
            self.defense = RotateDefense(angle_limit=5)
        elif defense_type in ['upscale_nearest', 'upscale_bicubic', 'upscale_bilinear']:
            self.defense = UpscaleDefense(mode=defense_type[8:], upscale_factor=0.5)
        elif defense_type == 'real-esrgan':
            self.defense = RealESRGANDefense()
        elif defense_type == 'mprnet':
            self.defense = MPRNETDefense(self.device)
        elif defense_type == 'diffpure':
            self.defense = DiffPureDefense(self.device)
        elif defense_type == 'diffpure100':
            self.defense = DiffPureDefenseT(100, self.device)
        elif defense_type == 'diffpure150':
            self.defense = DiffPureDefenseT(150, self.device)
        elif defense_type == 'diffpure200':
            self.defense = DiffPureDefenseT(200, self.device)
        elif defense_type == 'diffpure250':
            self.defense = DiffPureDefenseT(250, self.device)
        elif defense_type == 'diffpure300':
            self.defense = DiffPureDefenseT(300, self.device)
        elif defense_type == 'diffpure350':
            self.defense = DiffPureDefenseT(350, self.device)
        elif defense_type == 'diffpure400':
            self.defense = DiffPureDefenseT(400, self.device)
        else:
            self.defense = lambda x: x

    def forward(self, image, inference=False, defense=False, path=None):
        if defense:
            if 'diffpure' in self.defense_type:
                image = self.defense(image, path=path)
            else:
                image = self.defense(image)

        out = (
            self.model(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image))[
                -1
            ]
            * self.k
            + self.b
        )
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out

