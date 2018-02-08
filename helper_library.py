import cv2

def sliding_window(image, window, classifier)
	'''
		takes image, slides over by param window size, extracts features and classifies using param classifier
		
		param input:
			image: type np.array
			window: type int
			classifier: trained classifier object
			
		param output:
			pos_example: list of lists containing coordinates of positive examples
		
	'''
	assert isinstance(image, np.array)
	assert isinstance(window, int)
	
	Padded = cv2.copyMakeBorder(image,25,25,25,25,cv2.BORDER_REFLECT)
	[y,x,c] = image.shape
	pos_example = []

	print 'Scanning Image'
	for i in range(0,x,2):
		if(i%50==0):
			print '... ',
		for j in range(0,y,2):
			Test_Im = Padded[j:j+50,i:i+50]

			'''
				feature extraction of window
				classification based on features, trained classifier
			'''
			if(classification = True):
				pos_example.append([j,j+50,i,i+50])

	return pos_example
		








