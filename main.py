import cv2
import random
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import os ,sys
import glob
import h5py
import deepdish as dd
import time 
#ast.literal_eval(s)

#CV hw4 task2  0656100

PLOT = False
LOAD = True
SAVE = False
train_imgs = []
train_path = "./hw4_data/train"
test_path = "./hw4_data/test"

#h5py
#data = h5py.File("data.hdf5","w")
#=f.create_dataset("dset1", (20,), 'i')

#Kmeans
K= 50


if LOAD is True:
    
    #f = h5py.File("data.hdf5","w")

    train_data_dir = os.path.join(train_path,'*')
    files = glob.glob(train_data_dir)
    imlist = {}
    feature_list = {}
    count=0
    for each in glob.glob(train_path + "/*"):
        word = each.split("/")[-1]
        print ("Reading image category",word) 
        imlist[word] = []
        for imagefile in glob.glob(train_data_dir+word+"/*"):
            #print ("Reading file ", imagefile)
            im = cv2.imread(imagefile, 0)
            im = cv2.resize(im,(256,256), interpolation = cv2.INTER_CUBIC )
            imlist[word].append(im)
            count +=1 

    print("Done loading training data ",count,"images")
    
    sift = cv2.xfeatures2d.SIFT_create()    
    keypoint={}
    description = {}
    
    for cate ,images in imlist.items():
        print("SIFT now cate is ",cate)
        description[cate]=[]
        for img in images: 
            _ , des = sift.detectAndCompute(img,None)          #des (-1,128)
            description[cate].append(des)
    
    print("Done SIFT")

    all_des = None
    for cate ,des in description.items():
        print("len of description in ",cate)
        print(len(description[cate]))
        feature_list[cate]=[]
        for des in description[cate]:
            des = np.array(des)
            #print(des.shape)  # -1 128
            feature_list[cate].append(des)
            
            if (all_des is None):
                all_des = des 
            else:
                all_des = np.vstack((all_des,des))

    print(all_des.shape) #(779676,128)
    
    

    start_time = time.time()
    print("start kmeans")
    voc, variance = kmeans(all_des, K , 1 )
    #voc = KMeans(K).fit_predict(all_des) 
    print("Done kmeans {:.4f}".format(time.time()-start_time))
    #print(voc.shape)
    #print(variance)
       

    if SAVE == True:
        print("saving to image_data.h5")
        ff ={'img':imlist,'sift':all_des,'voc':voc,'variance':variance,'feature':feature_list}
        dd.io.save('image_data.h5', ff)
  


    img_histogram = np.zeros((1500, K), "float32")
    

    j=0 
    label_idx = 0
    label = np.zeros((1500,1),"float32")
    for cate in feature_list:
        
        for i in range(len(feature_list[cate])):
            words , distance = vq (feature_list[cate][i] , voc )
            label[j]= label_idx
            
            #print(words)
            for w in words:
                img_histogram[j][w] += 1
            j+=1
    
        label_idx+=1            


else:
 
    
    imlist, sift , voc ,img_feature= dd.io.load('image_data.h5', ['/img', '/sift','/voc','/feature' ])
    print("Done loading from image_data.h5")
    #print(voc.shape) #100,128

    # Calculate the histogram of features
    
    img_histogram = np.zeros((1500, K), "float32")
    

    j=0 
    label_idx = 0
    label = np.zeros((1500,1),"float32")
    for cate in img_feature:
        
        for i in range(len(img_feature[cate])):
            words , distance = vq (img_feature[cate][i] , voc )
            label[j]= label_idx
            
            #print(words)
            for w in words:
                img_histogram[j][w] += 1
            j+=1
    
        label_idx+=1            

    #print(img_histogram.shape)#1500,100
''' 
# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (img_histogram > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*1500+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(img_histogram)
img_histogram = stdSlr.transform(img_histogram)
'''

test_data_dir = os.path.join(test_path,'*')
files = glob.glob(test_data_dir)
test_data = {}
test_label = []
test_label_idx = 0
descrip = {}
test_data_count = 0 
for each in glob.glob(test_path + "/*"):
    word = each.split("/")[-1]
    print ("Reading Test image category",word)
    test_data[word] = []  
    for imagefile in glob.glob(test_data_dir+word+"/*"):
        im = cv2.imread(imagefile, 0)
        im = cv2.resize(im,(256,256), interpolation = cv2.INTER_CUBIC )
        test_data[word].append(im)
        test_data_count+=1

print("test data count ",test_data_count)
sift = cv2.xfeatures2d.SIFT_create()    
  
for cate ,images in test_data.items():
    descrip[cate]=[]
    for img in images: 
        _ , des = sift.detectAndCompute(img,None)          #des (-1,128)
        descrip[cate].append(des)

j=0 
test_histo = np.zeros((test_data_count,K),"float32")
for cate in descrip :
    for i in range(len(descrip[cate])):
        words , distance = vq (descrip[cate][i] , voc )
        test_label.append(test_label_idx)
        #print(words)
        for w in words:
            test_histo[j][w] += 1
        j+=1

    test_label_idx+=1            
'''
print("test histo",test_histo)

test_histo = stdSlr.transform(test_histo)

print("test histo",test_histo)
'''


#KNN 
knn = cv2.ml.KNearest_create()
knn.train(img_histogram,cv2.ml.ROW_SAMPLE,label)


for j in range(1,16):
    ret, results, neighbours ,dist = knn.findNearest(test_histo, j)
    print("knn: ",j)
    correct=0
    for i  in range(len(test_label)):
        if test_label[i] == results[i][0]:  
            correct+=1 

    #print(results)

    print("correct",correct,"/150")
    accuracy = correct*100.0/results.size
    print (accuracy)





if PLOT :
    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.set_title('key pointA')  
    ax1.imshow(imgA)        
    ax2 = plt.subplot2grid((3,2),(0,1),colspan=2)
    ax2.set_title('key pointB')  
    ax2.imshow(imgB) 
    ax3 = plt.subplot2grid((3,2),(1,0),colspan=2,rowspan=2)
    ax3.set_title('key match')
    ax3.imshow(img3)
    plt.show()
    
    
    ax1 = fig.add_subplot(3,2,1)
    ax1.imshow(img5,cmap="gray")
    ax1.set_title('epipolar lines')

    ax2 = fig.add_subplot(3,2,2)
    ax2.imshow(img2)
    ax2.set_title('keypoint')

    ax3 = fig.add_subplot(3,2,3)
    ax3.imshow(imgMatch)
    ax3.set_title('matches')

    plt.show()
    #cv2.imwrite('sift_keypoints.jpg',)    
    #fig.savefig('Result.png', bbox_inches='tight')








