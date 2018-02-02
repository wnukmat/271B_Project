import sys, cv2, os
import numpy as np
import matplotlib.pyplot as plt

def get_positive_sample(mask, image):
	Padded_mask = cv2.copyMakeBorder(mask,25,25,25,25,cv2.BORDER_CONSTANT)
	Padded_image = cv2.copyMakeBorder(image,25,25,25,25,cv2.BORDER_CONSTANT)
	index = np.where(Padded_mask[:,:,0] == 255)
	max_row = index[0].max()
	min_row = index[0].min()
	max_col = index[1].max()
	min_col = index[1].min()
	sample = Padded_image[min_row-5:max_row+5, min_col-5:max_col+5]
	
	return sample


folder = 'stage1_train'
positive_sample_folder = 'positive_samples'
try:
    os.stat(positive_sample_folder)
except:
    os.mkdir(positive_sample_folder) 
	
for file in os.listdir(folder):	
	for im in os.listdir(folder + "/" + file + "/images"):						
		if im.endswith(".png"):						
			img = cv2.imread(folder + "/" + file + "/images" + "/" + im)
			
		for im_mask in os.listdir(folder + "/" + file + "/masks"):						
			if im.endswith(".png"):
				mask = cv2.imread(folder + "/" + file + "/masks" + "/" + im_mask)
				sample = get_positive_sample(mask, img)
				cv2.imwrite(positive_sample_folder + '/' + im_mask, sample)