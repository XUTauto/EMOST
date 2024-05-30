from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import glob
import numpy as np
import os
import time
import re
from natsort import natsorted
from utils import LEGM
parser = argparse.ArgumentParser(description='HJ')
parser.add_argument('--cuda', action='store_true', help='use cuda', default='true')
parser.add_argument('--model', type=str, default=r'.\model\model_epoch_10.pth', help='model file to use')
parser.add_argument('--mri_file', type=str, default='img_test/BLK', help='the test MRI dataset path')
parser.add_argument('--pet_file', type=str, default='img_test/COL', help='the test PET dataset path')
opt = parser.parse_args()
print(opt)

def pretreatment(img):
	w, h = img.size
	new_w = w - w % 8
	new_h = h - h % 8
	l = (w - new_w) / 2
	top = (h - new_h) / 2
	r = (w + new_w) / 2
	bottom = (h + new_h) / 2
	return img.crop((l, top, r, bottom))

def color_conversion(out, cb, cr):
	img_y = out.data[0].numpy()
	img_y = img_y
	img_y *= 255.0
	img_y = img_y.clip(0, 255)
	img_y = Image.fromarray(np.uint8(img_y[0]), mode='L')
	img_cb = cb.resize(img_y.size, Image.BICUBIC)
	img_cr = cr.resize(img_y.size, Image.BICUBIC)
	img = Image.merge('YCbCr', [img_y, img_cb, img_cr]).convert('RGB')
	return img

def main():
	mri_address = opt.mri_file
	pet_address = opt.pet_file
	mri = sorted(glob.glob(os.path.join(mri_address, '*.jpg')), key=lambda x: int(re.findall(r'\d+', x)[0]))
	pet = sorted(glob.glob(os.path.join(pet_address, '*.jpg')), key=lambda x: int(re.findall(r'\d+', x)[0]))
	mri = natsorted(mri)
	pet = natsorted(pet)
	model = torch.load(opt.model)
	if opt.cuda:
		model = model.cuda()
	for i in range(len(mri)):
		MRI = mri[i]
		IMG_ONE = Image.open(MRI).convert('YCbCr')
		PET = pet[i]
		IMG_TWO = Image.open(PET).convert('YCbCr')
		IMG_ONE = pretreatment(IMG_ONE)
		IMG_TWO = pretreatment(IMG_TWO)
		y1, cb1, cr1 = IMG_ONE.split()
		y2, cb2, cr2 = IMG_TWO.split()
		Y1 = y1
		Y2 = y2
		Y1 = Variable(ToTensor()(Y1)).view(1, -1, Y1.size[1], Y1.size[0])
		Y2 = Variable(ToTensor()(Y2)).view(1, -1, Y2.size[1], Y2.size[0])
		if opt.cuda:
			Y1 = Y1.cuda()
			Y2 = Y2.cuda()
		with torch.no_grad():
			IMG1 = model.Extraction(Y1)
			IMG2 = model.Extraction(Y2)
			IMG = IMG1 * abs(IMG1 / (IMG2 + IMG1)) + IMG2 * abs(IMG2 / (IMG1+IMG2))
			IMG=0.5*IMG+0.5*LEGM(IMG1, IMG2)
			IMG = model.Reconstruction(IMG)
			IMG = IMG.cpu()
			IMG = color_conversion(IMG, cb2, cr2)
			result_file = os.path.join('result', '{}.jpg'.format(i+1))
			IMG.save(result_file)
if __name__ == '__main__':
	start = time.time()
	main()
	end = time.time()
	print(end - start)









