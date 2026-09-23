from torchvision.transforms import transforms, InterpolationMode
import torchvision.models as models

import torch.nn as nn

from mmpretrain import FeatureExtractor

import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
from glob import glob
import torch
from abc import ABC, abstractmethod
import os


class BaseFeatureExtractor(ABC):
    def __init__(self, embedding_size, device, weights=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f'Feature Extractor running on {self.device}')
        self.embedding_size = embedding_size
        self.encoder = self._init_model(weights)
        self.transform = transforms.Compose([
            transforms.Resize(235, interpolation=InterpolationMode.BICUBIC),
            # transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _image_preprocess(self, image):
        if isinstance(image,np.ndarray):
            # todo: check if bgr2rgb is necessary
            image = Image.fromarray(image) # function assumes a PIL image
        processed_image = image.copy()
        width, height = processed_image.size
        if height > width:
            processed_image = processed_image.rotate(90)
            # processed_image = processed_image.rotate(270)
        processed_image = transforms.Grayscale(3)(processed_image)
        return processed_image

    @abstractmethod
    def _extract_features(self, img):
        pass

    @abstractmethod
    def _init_model(self, weights):
        pass

    def extract(self, img):
        img = self._image_preprocess(img)
        return self._extract_features(img)

    def persist_features(self, features, feature_dir, paths):
        os.makedirs(feature_dir, exist_ok=True)
        for fv, path in tqdm(zip(features, paths)):
            feature_fn = path.replace('.jpg','.pt')
            feature_path = os.path.join(feature_dir, feature_fn)
            torch.save(fv, feature_path)

    def extract_and_persist_features_batched(self, image_dir, feature_dir, batch_size=100):
        image_paths = glob(f'{image_dir}/*.jpg')
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            batch_paths = image_paths[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            images = [(Image.open(img_path), img_path) for img_path in batch_paths]
            feature_vectors = []
            paths = []

            for image, image_path in images:
                fv = self.extract(image)
                feature_vectors.append(fv.detach().to('cpu'))
                paths.append(os.path.basename(image_path))

            self.persist_features(feature_vectors,feature_dir, paths)
        print(f'All features persisted to {feature_dir}.')



class TorchvisionFeatureExtractor(BaseFeatureExtractor):

    def _init_model(self, weights):
        model = models.resnet50(pretrained=True)
        if weights:
            checkpoint = torch.load(weights)
            backbone_weights = OrderedDict()
            for k,v in checkpoint.items():
                if k.startswith('base'):
                    newkey = k.split('base.')[-1]
                    backbone_weights[newkey] = v
            backbone_weights['fc.weight'] = torch.zeros((1000,2048))
            backbone_weights['fc.bias'] = torch.zeros((1000))
            model.load_state_dict(backbone_weights)
        model.eval().to(self.device)
        return model

    def _extract_features(self, img):
        t_img = self.transform(img).unsqueeze(0).to(self.device)
        my_embedding = torch.zeros(self.embedding_size)

        def copy_data(m, i, o):
            my_embedding.copy_(o.flatten())

        handle = self.encoder._modules.get("avgpool").register_forward_hook(copy_data)
        with torch.no_grad():
            self.encoder(t_img)
        handle.remove()
        return torch.unsqueeze(my_embedding, dim=0)


class MMPretrainFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, embedding_size, device, weights=None):
        torch.backends.cudnn.benchmark = True
        self._avgpool = nn.AdaptiveAvgPool2d((1,1))
        super().__init__(embedding_size, device, weights)

    def _init_model(self, weights):
        dev = str(self.device) if isinstance(self.device, torch.device) else self.device
        if weights:
            model = FeatureExtractor('resnet50_8xb32_in1k', pretrained=weights, device=dev)
        else:
            model = FeatureExtractor('resnet50_8xb32_in1k', device=dev)
        return model

    @torch.inference_mode()
    def _extract_features(self, img):
        img = np.array(img)  # FeatureExtractor handles preprocessing internally
        if self.device.type == "cuda":
            with torch.cuda.amp.autocast():  # half precision on GPU
                feature_maps = self.encoder(img, stage='backbone')[0][0]  # (C,H,W) on GPU
        else:
            feature_maps = self.encoder(img, stage='backbone')[0][0]
        pooled = self._avgpool(feature_maps).view(-1, feature_maps.size(0))
        return pooled




def torchvision_resnet50_extractor(device=None, weights=None):
    return TorchvisionFeatureExtractor(embedding_size=2048, device=device, weights=weights)


def mmpretrain_resnet50_extractor(device=None, weights=None):
    return MMPretrainFeatureExtractor(embedding_size=2048, device=device, weights=weights)

def odor_pretrained_extractor(device=None):
    weights = '/media/auction-catalogues/models/rn50_odor.pth'
    return mmpretrain_resnet50_extractor(device, weights)

def poses_pretrained_extractor(device=None):
    weights = '/media/auction-catalogues/models/rn50_poses.pth'
    return mmpretrain_resnet50_extractor(device, weights)