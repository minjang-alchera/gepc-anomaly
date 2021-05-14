import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .network.model import Net
from .config import config

class Feature_Extractor(object):
    def __init__(self):
        self.net = Net(reid=True)
        state_dict = torch.load(config.model_path)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(config.model_path))
        self.net.to(config.device)
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, imgs):
        tensor = []
        for img in imgs:
            tensor.append(self.norm(
                torch.from_numpy(cv2.resize(img, config.input_size)).to(config.device).float().permute(2, 0, 1) / 255.0).unsqueeze(0))
        tensor = torch.cat(tensor, 0)
        with torch.no_grad():
            feature = self.net(tensor)
            feature = feature.detach().cpu().numpy()
            features = []
            for n in range(feature.shape[0]):
                features.append(feature[n])
        return features

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

