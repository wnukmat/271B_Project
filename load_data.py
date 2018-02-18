import cv2, os
import numpy as np
import tensorflow as tf

class training_set():
	'''
	Loads training set from positive_examples, negative_examples folders
	creates class attributes for ease of use in batch training
	'''
	
	def __init__(self):
		self.batch_size = 100
		self.batch = 0

		
	def load_set(self, positive_sample_folder = None, negative_sample_folder = None):
		positive_sample_folder = '/home/matthew/271B_Project/positive_samples'
		negative_sample_folder = '/home/matthew/271B_Project/negative_samples'
		test_set_positive_folder = '/home/matthew/271B_Project/Validation_Set_Pos'
		test_set_negative_folder = '/home/matthew/271B_Project/Validation_Set_Neg'

		self.samples = []
		self.labels = []
		self.test_samples = []
		self.test_labels = []
		for im in os.listdir(positive_sample_folder):						
			if im.endswith(".png"):						
				img = cv2.imread(positive_sample_folder + "/" + im)
				self.samples.append(img/1.)
				self.labels.append([0,1])

		for im in os.listdir(negative_sample_folder):						
			if im.endswith(".png"):						
				img = cv2.imread(negative_sample_folder + "/" + im)
				self.samples.append(img/1.)
				self.labels.append([1,0])

		for im in os.listdir(test_set_positive_folder):						
			if im.endswith(".png"):						
				img = cv2.imread(test_set_positive_folder + "/" + im)
				self.test_samples.append(img/1.)
				self.test_labels.append([0,1])

		for im in os.listdir(test_set_negative_folder):						
			if im.endswith(".png"):						
				img = cv2.imread(test_set_negative_folder + "/" + im)
				self.test_samples.append(img/1.)
				self.test_labels.append([1,0])

		# Convert the data to float32 used in tf.nn.conv2
		self.samples = np.float32(self.samples)
		self.labels = np.array(self.labels)
		[self.num_examples, self.width, self.height, self.channels] = self.samples.shape

		# Randomize Order for training
		shuffle = np.random.permutation(len(self.samples))
		self.samples = self.samples[shuffle]
		self.labels = self.labels[shuffle] 

		print 'Data Set Loaded'

	def next_batch(self):
		start = self.batch_size*self.batch
		end = start + self.batch_size

		if(self.num_examples > end):
			sample_batch = self.samples[start:end]
			label_batch = self.labels[start:end]
			self.batch = self.batch + 1
		else:
			sample_batch = self.samples[start:-1]
			label_batch = self.labels[start:-1]
			self.batch = 0

		return sample_batch, label_batch			






