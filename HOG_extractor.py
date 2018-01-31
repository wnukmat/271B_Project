from sklearn import svm
from sklearn.externals import joblib

def feature_extractor(im):
	'''
	Histogram of Gradient features extracted from input image
	expects a 50x50 pixel image as input
	'''
	winSize = (50,50)		# corresponding to the size of the input image choosen
	blockSize = (20,20)		# 2x Cellsize, parameter handling illumination variations
	blockStride = (10,10)	# 50% Blocksize, normalization factor	
	cellSize = (10,10)		# dimensionality reduction factor to capture highly informative features
	nbins = 9				# default recogmendation of N. Dalal
	 
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
		
	return hog.compute(im.astype(np.uint8))
	
	
def train_SVM(samples, labels):
	# Create SVM Classifier
	clf = svm.SVC()
	clf.fit(samples, labels) 
	joblib.dump(clf, 'trained_phone_finder.pkl', compress=9)
	# print 'error: ', 1-clf.score(samples, labels)
	
	return clf
