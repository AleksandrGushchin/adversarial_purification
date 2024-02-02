import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
import cv2

class MPRNETDefense:
	model = None

	def load_checkpoint(self, model, weights):
	    checkpoint = torch.load(weights)
	    try:
	        self.model.load_state_dict(checkpoint["state_dict"])
	    except:
	        state_dict = checkpoint["state_dict"]
	        new_state_dict = OrderedDict()
	        for k, v in state_dict.items():
	            name = k[7:] # remove `module.`
	            new_state_dict[name] = v
	        self.model.load_state_dict(new_state_dict)

	def __init__(self, device):
		task = "Denoising"

		# Load corresponding model architecture and weights
		load_file = run_path("defenses/transforms/modules/MPRNet.py")
		self.model = load_file['MPRNet']()
		self.model.to(device)

		weights = "defenses/transforms/weights/mprnet_denoise.pth"
		self.load_checkpoint(self.model, weights)
		self.model.eval()

	def __call__(self, image):
		img_multiple_of = 8

		# print(image)
		# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.squeeze(0)#permute(0, 2, 3, 1)
		img = image
		input_ = image
		# input_ = TF.to_tensor(img).unsqueeze(0).cuda()

		# Pad the input if not_multiple_of 8
		h,w = input_.shape[1], input_.shape[2]
		H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
		padh = H-h if h%img_multiple_of!=0 else 0
		padw = W-w if w%img_multiple_of!=0 else 0
		# print(h,w)
		# print(H,W)
		# print(padh, padw)
		input_ = F.pad(input_, (0,padw,0,padh), 'reflect').unsqueeze(0)


		# print(input_.shape)
		with torch.no_grad():
		    restored = self.model(input_)
		restored = restored[0]
		restored = torch.clamp(restored, 0, 1)

		# Unpad the output
		restored = restored[:,:,:h,:w]

		# print("restored", restored.shape)

		return restored


        # image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        # transform = ....
        # image = transform(image=image)['image']
        # image = torch.from_numpy(image).cuda()
        # image = image.unsqueeze(0).permute(0, 3, 1, 2)
