import cv2, os
import numpy as np

def badPath(path):
    try:
        os.stat(path)
        return False
    except:
        return True


# Set all folders before using
folder = '/Users/Zaki/Documents/UCSD Winter 18/ECE271B/Nuclei/data/stage1_train/'
positive_sample_folder = '/Users/Zaki/Documents/UCSD Winter 18/ECE271B/Nuclei/data/positive_samples'
#positive_sample_folder = '/Users/Zaki/Documents/UCSD Winter 18/ECE271B/Nuclei/data/positive_true_samples'
negative_sample_folder = '/Users/Zaki/Documents/UCSD Winter 18/ECE271B/Nuclei/data/negative_samples'

try:
    os.stat(positive_sample_folder)
except:
    raise
#    os.mkdir(positive_sample_folder) 
    
try:
    os.stat(negative_sample_folder)
except:
    raise
#    os.mkdir(negative_sample_folder) 

image_count = 0
trueSize = False #option to output postive samples of true size
desiredS = 100
pad = 5
negativeThresh = 0.15
for file in os.listdir(folder):
    if badPath(folder + "/" + file + "/images"):
        continue
    
    for im in os.listdir(folder + "/" + file + "/images"):
        if badPath(folder + "/" + file + "/masks"):
            continue
        
        img = cv2.imread(folder + "/" + file + "/images" + "/" + im)
        
        image_count += 1
        if image_count%100 == 0:
            print image_count
        totalMask = np.zeros((img.shape[0],img.shape[1])).astype('uint8')
        mask_count = 0
        for im_mask in os.listdir(folder + "/" + file + "/masks"):
            if im.endswith(".png"):
                mask = cv2.imread(folder + "/" + file + "/masks" + "/" + im_mask,cv2.IMREAD_GRAYSCALE)
                mask_count += 1
                totalMask += cv2.threshold(mask,1,1,cv2.THRESH_BINARY)[1]
                countours = cv2.findContours(mask,1,2)
                cnt = countours[0]
                x,y,w,h = cv2.boundingRect(cnt)
                aspect = [w,h]
                s = max(aspect)
                sample = np.zeros((s+2*pad,s+2*pad,3))
                s_y_start = 0
                s_x_start = 0
                s_y_end = sample.shape[0]
                s_x_end = sample.shape[1]    
                xpad = pad
                ypad = pad
                if y + s + pad > img.shape[0]:
                    s_y_end = img.shape[0] - (y + s + pad)
                if x + s + pad > img.shape[1]:
                    s_x_end = img.shape[1] - (x + s + pad)
                if y-pad < 0:
                    s_y_start = pad - y
                    ypad = pad - s_y_start
                if x-pad < 0:
                    s_x_start = pad - x
                    xpad = pad - s_x_start
                sample[s_y_start:s_y_end,s_x_start:s_x_end] = img[y-ypad:y+s+pad,x-xpad:x+s+pad]
                if not trueSize:
                    if (s + 2*pad) <= desiredS:
                        newSamp = np.zeros((desiredS,desiredS,3))
                        sampS = sample.shape[0]
                        newSamp[desiredS/2 - sampS/2:desiredS/2 - sampS/2 + sampS,desiredS/2 - sampS/2:desiredS/2 - sampS/2 + sampS] = sample
                        sample = newSamp
                    else:
                        newSamp = np.zeros((desiredS,desiredS,3))
                        cv2.resize(sample,(desiredS,desiredS), newSamp)
                        sample = newSamp
                        
                cv2.imwrite(positive_sample_folder + '/' + im_mask, sample)
        
        negs = 0
        i = 0
        j = 0
        negsLim = False
        ijump = False
        while i < totalMask.shape[0]-desiredS:
            while j < totalMask.shape[1]-desiredS:
                if negs == mask_count:
                    negsLim = True
                    break
                val = sum(sum(totalMask[i:i+desiredS,j:j+desiredS]))
                if val/(float(desiredS)**2) <= 0.15:
                    negs += 1
                    neg_sample = totalMask[i:i+desiredS,j:j+desiredS]
                    cv2.imwrite(negative_sample_folder + '/' + str(negs) + '_' + im, neg_sample)
                    j += desiredS
                    if j >= totalMask.shape[1]-desiredS:
                        ijump = True
                    continue
                j += 1
            
            if negsLim:
                break
            if ijump:
                i += desiredS
                ijump = False
            else:
                i += 1
            j = 0