"""
Author: Joseph Mattern
File: generate_features.py
Python Version: 2.7
Description:
    Performs actions based on the control flow parameters of the code
    If CREATE_MASKS is set:
        The individual masks within the training set will be combined into a 
        single mask per image
    If GEN_DESCS is set:
        Each image in the training set will be put through the feature
        detectors and the descriptors will be saved to a separate file per 
        feature detector
    If FIT_DESCS is set:
        The feature descriptors are loaded and each one is put through an SVM.
        The SVMs are then saved to pickle files for each descriptor
    If CLASSIFY_VALID is set:
        The validation images are put through the feature detectors and the 
        keypoints are saved as their appropriate labels.  The labeled keypoints
        are then saved to a pickle file.
    If CLUSTER_VALID is set:
        The labeled leypoints are loaded and put through an adaptive k-means
        clustering to find regions of interest.  These regions are then filled
        and saved as a mask for every image.
    If CLASSIFY_TEST is set:
        The test images are put through the feature detectors and the keypoints
        are saved as their appropriate labels.  The labeled keypoints are then 
        saved to a pickle file.
    If CLUSTER_TEST is set:
        The labeled leypoints are loaded and put through an adaptive k-means
        clustering to find regions of interest.  These regions are then filled
        and saved as a mask for every image.
"""
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.ndimage.morphology import binary_fill_holes
import cPickle as pickle


#Control flow parameters
CREATE_MASKS = False
GEN_DESCS = False
FIT_DESCS = False
CLASSIFY_VALID = False
CLUSTER_VALID = True

classifier_total = list(["orb","surf","sift"]) #DO NOT CHANGE
classifier_list = list(["orb","surf","sift"]) #Change to affect which classifiers are used

# Set all folders before using
folder = "data/training_data"
test_folder = "data/test_data"
test_results = "results"
working_folder = "working"
mask_folder = "data/comb_masks"
#posKps = list()
#negKps = list()

#Path Checking Function
def badPath(path):
    try:
        os.stat(path)
        return False
    except:
        return True

#load training, test, and validation set file names
train_list = list()
valid_list = list()
test_list = list()
with open("data/train.txt") as f:
    train_list = f.readlines()
    for i,s in enumerate(train_list):
        file,txt = s.split('.')
        train_list[i] = file
with open("data/valid.txt") as f:
    valid_list = f.readlines()
    for i,s in enumerate(valid_list):
        file,txt = s.split('.')
        valid_list[i] = file
with open("data/test.txt") as f:
    test_list = f.readlines()
    for i,s in enumerate(test_list):
        file,txt = s.split('.')
        test_list[i] = file


#Create the combinted masks for each training image given the segmented masked
if CREATE_MASKS:
    for file in os.listdir(folder):
        if badPath(folder + "/" + file + "/images"):
            continue
        comb_mask = np.array([0])
        for mask in os.listdir(folder + "/" + file + "/masks"):
            if len(comb_mask) == 1:
                comb_mask = cv2.imread(folder + "/" + file + "/masks" + "/" + mask,cv2.IMREAD_GRAYSCALE)
            else:
                comb_mask = comb_mask + cv2.imread(folder + "/" + file + "/masks" + "/" + mask,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(mask_folder + "/" + file+ ".png", comb_mask)
    print "Composite Masks Created and Saved to " + mask_folder

#Generate the descriptors and keypoints for each image using the listed feature detectors
if GEN_DESCS:
    posDescsDict = dict()
    negDescsDict = dict()
    for classifier in classifier_list:
        posDescs = np.zeros((1,1))
        negDescs = np.zeros((1,1))
        for file in os.listdir(folder):
            if file not in train_list:
                continue
            comb_mask = cv2.imread(mask_folder + "/" + file + ".png",cv2.IMREAD_GRAYSCALE)
            for im in os.listdir(folder + "/" + file + "/images"):
                if badPath(folder + "/" + file + "/masks"):
                    continue
                img = cv2.imread(folder + "/" + file + "/images" + "/" + im,cv2.IMREAD_GRAYSCALE)
                if classifier == 'orb':
                    orb = cv2.ORB_create()
                    kps, descs = orb.detectAndCompute(img,None)
                elif classifier == 'surf':
                    surf = cv2.xfeatures2d.SURF_create()
                    kps, descs = surf.detectAndCompute(img,None)
                elif classifier == 'sift':
                    sift = cv2.xfeatures2d.SIFT_create()
                    kps, descs = sift.detectAndCompute(img,None)
                for i,kp in enumerate(kps):
                    pt = kp.pt
                    desc = descs[i,:]
                    desc = np.reshape(desc,(1,len(desc)))
                    if comb_mask[int(pt[1]),int(pt[0])] > 0:
#                        posKps.append(kp)
                        if posDescs.shape[1] == 1:
                            posDescs = desc
                        else:
                            posDescs = np.vstack((posDescs,desc))
                    else:
#                        negKps.append(kp)
                        if negDescs.shape[1] == 1:
                            negDescs = desc
                        else:
                            negDescs = np.vstack((negDescs,desc))
        if posDescs.shape[0] > negDescs.shape[0]:
            while posDescs.shape[0] > negDescs.shape[0]:
                negDescs = np.vstack((negDescs,negDescs[i]))
                i = i + 1
                if i == negDescs.shape[0]:
                    i = 0
            
        posDescsDict[classifier] = posDescs
        negDescsDict[classifier] = negDescs
        print classifier + " Descriptors Generated"
        
#Save or load the descriptors
if GEN_DESCS:
    for classifier in classifier_list:
        np.savetxt(working_folder + "/training_descriptors/" + classifier + "_pos_descs.txt", posDescsDict[classifier], delimiter=',')
        print "Positive " +classifier + " Descriptors Saved to: " + working_folder + "/training_descriptors/" + classifier + "_pos_descs.txt"
        np.savetxt(working_folder + "/training_descriptors/" + classifier + "_neg_descs.txt", negDescsDict[classifier], delimiter=',')
        print "Negative " +classifier + " Descriptors Saved to: " + working_folder + "/training_descriptors/" + classifier + "_neg_descs.txt"
elif FIT_DESCS:
    posDescsDict = dict()
    negDescsDict = dict()
    for classifier in classifier_list:
        posDescsDict[classifier] = np.loadtxt(working_folder + "/training_descriptors/" + classifier + "_pos_descs.txt", delimiter=',')
        print "Positive " +classifier + " Descriptors Loaded from: " + working_folder + "/training_descriptors/" + classifier + "_pos_descs.txt"
        negDescsDict[classifier] = np.loadtxt(working_folder + "/training_descriptors/" + classifier + "_neg_descs.txt", delimiter=',')
        print "Negative " +classifier + " Descriptors Loaded from: " + working_folder + "/training_descriptors/" + classifier + "_neg_descs.txt"
      
#Fit the descriptors using a SVM
if FIT_DESCS:
    svcDict = dict()
    for classifier in classifier_list:
        svcDict[classifier] = dict()
        X1 = posDescsDict[classifier]
        y1 = np.ones(posDescsDict[classifier].shape[0])
        X2 = negDescsDict[classifier]
        y2 = np.zeros(negDescsDict[classifier].shape[0])
        
        X = np.vstack((X1,X2))
        y = np.hstack((y1,y2))
        
        # Create the SVC model object
        C = 1.0 # SVM regularization parameter
        svcDict[classifier] = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X, y)
        print classifier + " SVM trained"

#Save or load the fitted svm object for each feature detector
if FIT_DESCS:
    with open(working_folder + '/feature_classifiers/learned_svms.dat', 'w') as file:
        pickle.dump(svcDict,file) # use `pickle.loads` to do the reverse
        classifiers = ""
        for classifier in svcDict.keys():
            classifiers = classifiers + classifier + " "
        print classifiers + "SVM(s) saved to: " + working_folder + '/feature_classifiers/learned_svms.dat'
elif CLASSIFY_VALID:
    with open(working_folder + '/feature_classifiers/learned_svms.dat', 'r') as file:
        svcDict = pickle.load(file) # use `pickle.loads` to do the reverse
        classifiers = ""
        for classifier in svcDict.keys():
            classifiers = classifiers + classifier + " "
        print classifiers + "SVM(s) loaded from: " +  working_folder + '/feature_classifiers/learned_svms.dat'
    

#Use the fitted SVMs to classify each keypoint in the validation image
if CLASSIFY_VALID:
    svcRes = dict()
    for classifier in classifier_list:
        svcRes[classifier] = dict()
        for file in os.listdir(folder):
            if file not in valid_list:
                continue
            svcRes[classifier][file] = dict()
            comb_mask = cv2.imread(mask_folder + "/" + file + ".png",cv2.IMREAD_GRAYSCALE)
            for im in os.listdir(folder + "/" + file + "/images"):
                img = cv2.imread(folder + "/" + file + "/images" + "/" + im,cv2.IMREAD_GRAYSCALE)
                if classifier == 'orb':
                    orb = cv2.ORB_create()
                    kps, descs = orb.detectAndCompute(img,None)
                elif classifier == 'surf':
                    surf = cv2.xfeatures2d.SURF_create()
                    kps, descs = surf.detectAndCompute(img,None)
                elif classifier == 'sift':
                    sift = cv2.xfeatures2d.SIFT_create()
                    kps, descs = sift.detectAndCompute(img,None)
                serialKpList = list()
                if not np.all(descs):
                    svcRes[classifier][file]['labels'] = list()
                    svcRes[classifier][file]['kps'] = list()
                    continue
                for kp in kps:
                    temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                    serialKpList.append(temp)
                svcRes[classifier][file]['labels'] = svcDict[classifier].predict(descs)
                svcRes[classifier][file]['kps'] = serialKpList
        print classifier + " Features Generated and Classified for Validation Set"

#save validation results    
if CLASSIFY_VALID:
    with open(test_results + '/valid_res.dat', 'w') as file:
        pickle.dump(svcRes,file) # use `pickle.loads` to do the reverse
        classifiers = ""
        for classifier in svcRes.keys():
            classifiers = classifiers + classifier + " "
        print classifiers + "Classified Features saved to: " + test_results + '/valid_res.dat'
   
#load and Deserialize validation results
if CLUSTER_VALID:
    with open(test_results + '/valid_res.dat', 'r') as file:
        svcResSerial = pickle.load(file) # use `pickle.loads` to do the reverse  
        svcRes = dict()
        for classifier in svcResSerial.keys():
            svcRes[classifier] = dict()
            for file in svcResSerial[classifier].keys():
                svcRes[classifier][file] = dict()
                svcRes[classifier][file]['labels'] = svcResSerial[classifier][file]['labels']
                kps = list()
                for serial_kp in svcResSerial[classifier][file]['kps']:
                    temp = cv2.KeyPoint(x=serial_kp[0][0],y=serial_kp[0][1],_size=serial_kp[1], _angle=serial_kp[2], _response=serial_kp[3], _octave=serial_kp[4], _class_id=serial_kp[5]) 
                    kps.append(temp)
                svcRes[classifier][file]['kps'] = kps
        classifiers = ""
        for classifier in svcRes.keys():
            classifiers = classifiers + classifier + " "
        print classifiers + "Classified Features loaded from: " + test_results + '/valid_res.dat'

#Cluster the classified data points
if CLUSTER_VALID:
    for classifier in classifier_list:
#        for file in ['53df5150ee56253fe5bc91a9230d377bb21f1300f443ba45a758bcb01a15c0e4']:
        for file in os.listdir(folder):
            if file not in valid_list:
                continue
            for im in os.listdir(folder + "/" + file + "/images"):
                gray = cv2.imread(folder + "/" + file + "/images" + "/" + im,cv2.IMREAD_GRAYSCALE)
                ##OLD FLOOD FILL SETUP
#                    gray_edge = cv2.Canny(gray,1,100)
#                    h, w = gray_edge.shape[:2]
#                    mask = np.zeros((h + 2, w + 2), np.uint8)
                
                # Draw the keypoints
                posKps = list()
                negKps = list()
                posKpLocs = list()
                negKpLocs = list()
                for i,kp in enumerate(svcRes[classifier][file]['kps']):
                    if svcRes[classifier][file]['labels'][i] == 1:
                        posKps.append(kp)
                        posKpLocs.append(np.array(kp.pt,dtype = np.uint32))
                    else:
                        negKps.append(kp)
                        negKpLocs.append(np.array(kp.pt,dtype = np.uint32))
                    
                    #Adaptive KMeans
                    MoreClusters = True
                    k = 1
                    while(MoreClusters == True):
                        # print 'Clustering: K = ' + str(k)
                        MoreClusters = False
                        k_means = KMeans(n_clusters=k)
                        k_means.fit(posKpLocs) 
                        
                        for i in range(len(k_means.labels_)):
                            dist = np.linalg.norm(posKpLocs[i]-k_means.cluster_centers_[k_means.labels_[i]] )
                            if(dist > 30):
                                MoreClusters = True
                        k += 1
                    majority_class = max(set(k_means.labels_), key=list(k_means.labels_).count)
                    
                    alpha = k_means.cluster_centers_[majority_class]
                        
                        ##OLD FLOOD FILL
#                    for kp in posKps:
#                        gray_edge_temp = np.array(gray_edge)
#                        cv2.floodFill(gray_edge_temp,mask,tuple(np.array(kp.pt,dtype = np.uint8)),255)
#                        a1 = np.sum(gray_edge_temp)
#                        a2 = np.sum(gray_edge)
#                        im_size = gray_edge.shape[0]*gray_edge.shape[1]
#                        if float(a1 - a2)/float(255*im_size) < 0.4:
#                            gray_edge = np.array(gray_edge_temp)
#                    im2, contours, hierarchy = cv2.findContours(gray_edge,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                    contourList = list()
#                    for i,kp in enumerate(posKps):
#                        kp_pt = np.array(kp.pt,dtype = np.uint32)
#                        for contour in contours:
#                            for pt in contour[0]:
#                                if np.array_equal(kp_pt,pt):
#                                    contourList.append(i-1)
                    
                    #Display Results
#                    edge_img = cv2.cvtColor(gray_edge, cv2.COLOR_GRAY2BGR)
#                    img_out = edge_img
#                    cv2.drawKeypoints(edge_img,posKps,img_out,color=(0,255,0), flags=0)
#                    cv2.drawKeypoints(edge_img,negKps,img_out,color=(0,0,255), flags=0)
#                    cv2.startWindowThread()
#                    cv2.namedWindow(classifier)
#                    cv2.imshow(classifier,img_out)
#                    cv2.waitKey(0)
#                    cv2.destroyAllWindows()
                                    
    