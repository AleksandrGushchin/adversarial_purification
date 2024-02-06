import os, cv2
from torchvision import transforms
from glob import glob

class DiffPureDefenseT:
    def __init__(self, t=400, device='cpu'):
        self.device = device
        self.t = t

    def __call__(self, image, path):
        path = path.replace('../../', '')
        diffpure_path = '/home/24a_guh@lab.graphicon.ru/pur/DiffPure/data/results/' + path
        diffpure_path = diffpure_path.replace('dataset_full', 'experiment1')
        diffpure_path = os.path.join(os.path.dirname(diffpure_path), f"t={self.t}_" + os.path.basename(path))
        if os.path.isfile(diffpure_path):
            # print(diffpure_path)
            image = cv2.imread(diffpure_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.ToTensor()
            image = transform(image).unsqueeze(0).to(self.device)
            return image
        else:
            print(f'not found! {diffpure_path}')
            return image
