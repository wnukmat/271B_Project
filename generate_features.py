import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def badPath(path):
    try:
        os.stat(path)
        return False
    except:
        return True

#Control flow parameters
CREATE_MASKS = False
GEN_DESCS = False
FIT_DESCS = True

classifier_total = list(["orb","surf","sift"]) #DO NOT CHANGE
classifier_list = list(["orb","surf","sift"])

# Set all folders before using
folder = "data/training"
test_folder = "data/test"
mask_folder = "data/comb_masks"
#posKps = list()
#negKps = list()
posDescsDict = dict()
negDescsDict = dict()
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
    
if GEN_DESCS:
    for classifier in classifier_list:
        posDescs = np.zeros((1,1))
        negDescs = np.zeros((1,1))
        for file in os.listdir(folder):
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
        posDescsDict[classifier] = posDescs
        negDescsDict[classifier] = negDescs
            
                                
    #img_c = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #img_out = img_c
    #cv2.drawKeypoints(img_c,kps,img_out,color=(0,255,0), flags=0)
    #cv2.imshow("Orb Positive",img_out)
    #cv2.waitKey()
    #
    #img_out = img
    #cv2.drawKeypoints(img_c,kps,img_out,color=(0,255,0), flags=0)
    #cv2.imshow("Orb Negative",img_out)
    #cv2.waitKey()   
                                
if GEN_DESCS:
    for classifier in classifier_list:
        np.savetxt("data/" + classifier + "_pos_descs.txt", posDescsDict[classifier], delimiter=',')
        np.savetxt("data/" + classifier + "_neg_descs.txt", negDescsDict[classifier], delimiter=',')
elif FIT_DESCS:
    for classifier in classifier_list:
        posDescsDict[classifier] = np.loadtxt("data/" + classifier + "_pos_descs.txt", delimiter=',')
        negDescsDict[classifier] = np.loadtxt("data/" + classifier + "_neg_descs.txt", delimiter=',')
     

if FIT_DESCS:
    svcDict = dict()
    svcRes = dict()
#    for classifier in classifier_list:
    for classifier in ['orb']:
        X1 = posDescsDict[classifier]
        y1 = np.ones(posDescsDict[classifier].shape[0])
        X2 = negDescsDict[classifier]
        y2 = np.zeros(negDescsDict[classifier].shape[0])
        
        X = np.vstack((X1,X2))
        y = np.hstack((y1,y2))
        
        # Create the SVC model object
        C = 1.0 # SVM regularization parameter
        svcDict[classifier] = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X, y)
        num = 0
        for file in os.listdir(test_folder):
            if not num:
                num = num+1
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
                res = svcDict[classifier].predict(descs)
            
        



   
                    
                
                

        
                    