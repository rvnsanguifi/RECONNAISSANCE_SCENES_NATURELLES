###################################################################################################################################
###           MODULE : RECONNAISSANCE DES FORMES                                                                               ####
###          PROJET: UNE APPLICATION SUR LA RECONNAISSANCE DES SCENES NATURELLES                                               ####
###           TRAVAIL REALISÉ PAR: NSANGU NGIMBI HERVE                                                                         ####
###           ETUDIANT EN MASTER 2 A L'INSTITUT FRANCOPHONE INTERNATIONAL                                                      ####
###                                                                                                                            ####
###                                                                                                                            ####
###                                                                                                                            ####
###################################################################################################################################

from urllib.request import urlretrieve
from os import listdir
from os.path import isfile, join, exists
from zipfile import ZipFile
##for random split of dataset into a training set and a test set ###
from sklearn.model_selection import train_test_split
## image processing routines for feature extraction/transformation##
from skimage.feature import daisy,hog
from skimage import io
from skimage.color import rgb2gray
import skimage
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import svm

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

#%matplotlib inline
#Download Dataset


if not exists('scene_categories/'):
    if not exists('scene_categories.zip'):
        print('Downloading scene_categories.zip')
        urlretrieve('http://www-cvr.ai.uiuc.edu/ponce_grp/data/scene_categories/scene_categories.zip', 'scene_categories.zip')
        print('Downloaded scene_categories.zip')
    print('Extracting scene_categories.zip')
    zipfile = ZipFile('scene_categories.zip', 'r')
    zipfile.extractall('./scene_categories')
    zipfile.close()
    print('Extracted scene_categories.zip')
else:
    print('Dataset already downloaded and extracted!')
    
#Dataset already downloaded and extracted!
#Get all the filenames (including the full path) in a folder as a list.

    
def get_filenames(path):
    onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

#Function to extract daisy features as well as hog features from an image

    
def extract_daisy_and_hog_features_from_image(file_path,daisy_step_size=32,daisy_radius=32,hog_pixels_per_cell=16,hog_cells_per_block=1):
    img = io.imread(file_path)
    img_gray = rgb2gray(img)
    img=skimage.transform.resize(img_gray,(300,250)) ##resize to a suitable dimension, avg size of images in the dataset
    #original, histograms=6
    descs = daisy(img, step=daisy_step_size, radius=daisy_radius, rings=2, histograms=8,orientations=8, visualize=False)
    #calculate daisy feature descriptors
    descs_num = descs.shape[0] * descs.shape[1]
    daisy_desriptors=descs.reshape(descs_num,descs.shape[2])
    hog_desriptor=hog(img, orientations=8, pixels_per_cell=(hog_pixels_per_cell, hog_pixels_per_cell),cells_per_block=(hog_cells_per_block, hog_cells_per_block), visualise=False,feature_vector=True)
    return daisy_desriptors,hog_desriptor


    
def plot_file(fname):
    img_data=plt.imread(fname)
    plt.imshow(rgb2gray(img_data),cmap='Greys_r')
    

    
base_path="scene_categories/" ##path where image data is kept
## take a descriptor sample to show the shape of arrays ##
fname="kitchen/image_0001.jpg"
plot_file(base_path+fname)
img_data=plt.imread(base_path+fname)
plt.imshow(rgb2gray(img_data),cmap='Greys_r')
daisy_sample,hog_sample=extract_daisy_and_hog_features_from_image(base_path+fname,daisy_step_size=8,daisy_radius=8)
print('DAISY descriptor size:',daisy_sample.shape)
print('HOG descriptor size:',hog_sample.shape)
#DAISY descriptor size: (1080, 136)
#HOG descriptor size: (2160,)



img_width=300
img_height=250
hog_pixels_per_cell=16
orientations=8



print('HOG vector size=',(img_width/hog_pixels_per_cell)*(img_height/hog_pixels_per_cell)*orientations)
#HOG vector size= 2160
#Load file names corresponding to each scene category into lists



category_names=listdir(base_path) ##
for i in range(len(category_names)):
    print(category_names[i],'=',i)
    
print('total categories:',len(category_names))
dataset_filenames=[] ##list to keep path of all files in the database
dataset_labels=[]
##category_names.index('store')  list the numeric representation of the category
##category_names[0] list the text representation of the category id
for category in category_names:
    category_filenames=get_filenames(base_path+category+"/")##get all the filenames in that category
    category_labels=np.ones(len(category_filenames))*category_names.index(category) ##label the category with its index position
    dataset_filenames=dataset_filenames+category_filenames
    dataset_labels=dataset_labels+list(category_labels)

print('total dataset size:',len(dataset_filenames))



train_filenames,test_filenames,train_labels,test_labels=train_test_split(dataset_filenames,dataset_labels,train_size=1300, stratify=dataset_labels)
print('total files in training split:',len(train_filenames))
print('total files in testing split:',len(test_filenames))



training_data_feature_map={} ##map to store daisy feature as well as hog feature for all training datapoints
daisy_descriptor_list=[] ##list to store all daisy descriptors to form our visual vocabulary by clustering
counter=0
for fname in tqdm(train_filenames):
    daisy_features,hog_feature=extract_daisy_and_hog_features_from_image(fname,daisy_step_size=8,daisy_radius=8)
    ###extract DAISY features and HOG features from the image and save in a map###
    training_data_feature_map[fname]=[daisy_features,hog_feature]
    daisy_descriptor_list=daisy_descriptor_list+list(daisy_features)

print('Total daisy descriptors:',len(daisy_descriptor_list))


def cluster_daisy_features(daisy_feature_list,number_of_clusters):
    #km=KMeans(n_clusters=number_of_clusters)
    km=MiniBatchKMeans(n_clusters=number_of_clusters,batch_size=number_of_clusters*10)
    km.fit(daisy_feature_list)
    return km
#In [14]:
### hide warnings ##
import warnings
warnings.filterwarnings('ignore')
#Number of clusters is set as 600#,takes aprox 10 mins to run on standard laptop
#In [15]:
daisy_cluster_model=cluster_daisy_features(daisy_descriptor_list,600) 
daisy_cluster_model.n_clusters


def extract_daisy_hog_hybrid_feature_from_image(fname,daisy_cluster_model):
    #incase if we have encountered the file during training, the daisy and hog features would already have been computed
    if fname in training_data_feature_map:
        daisy_features=training_data_feature_map[fname][0]
        hog_feature=training_data_feature_map[fname][1]
    else:
        daisy_features,hog_feature=extract_daisy_and_hog_features_from_image(fname,daisy_step_size=8,daisy_radius=8)
        
    ##find to which clusters each daisy feature belongs
    img_clusters=daisy_cluster_model.predict(daisy_features) 
    cluster_freq_counts=pd.DataFrame(img_clusters,columns=['cnt'])['cnt'].value_counts()
    bovw_vector=np.zeros(daisy_cluster_model.n_clusters) ##feature vector of size as the total number of clusters
    for key in cluster_freq_counts.keys():
        bovw_vector[key]=cluster_freq_counts[key]

    bovw_feature=bovw_vector/np.linalg.norm(bovw_vector)
    hog_feature=hog_feature/np.linalg.norm(hog_feature)
    return list(bovw_feature)+list(hog_feature)
#Training data feature extraction

#In [17]:

XTRAIN=[]
YTRAIN=[]
for i in tqdm(range(len(train_filenames))):
    XTRAIN.append(extract_daisy_hog_hybrid_feature_from_image(train_filenames[i],daisy_cluster_model))
    YTRAIN.append(train_labels[i])
    

hybrid_classifier=svm.SVC(C=10**1.6794140624999994, gamma=10**-0.1630955304365928, decision_function_shape='ovo') #cross-validated hyper-parameters
hybrid_classifier.fit(XTRAIN,YTRAIN)


plot_file(test_filenames[3])
print('true label:',test_labels[3])
hybrid_feature_vector=extract_daisy_hog_hybrid_feature_from_image(test_filenames[3],daisy_cluster_model)
print('prediction:',hybrid_classifier.predict([hybrid_feature_vector]))


### show a sample classification on external file ###

external_file_name="test/l1.jpg"
plot_file(external_file_name)
hybrid_feature_vector=extract_daisy_hog_hybrid_feature_from_image(external_file_name,daisy_cluster_model)
print('Prediction:',hybrid_classifier.predict([hybrid_feature_vector]))



XTEST=[]
YTEST=[]
for i in tqdm(range(len(test_filenames))):
    XTEST.append(extract_daisy_hog_hybrid_feature_from_image(test_filenames[i],daisy_cluster_model))
    YTEST.append(test_labels[i])

print('Hybrid Classifier Mertics')
print('No. of test instances:',len(XTEST),len(YTEST))
### Accuracy Report ###
hybridpred=hybrid_classifier.predict(XTEST)
print('Overall accuracy:',accuracy_score(YTEST,hybridpred))
print(classification_report(YTEST, hybridpred, target_names=category_names))
print('Confusion matrix:\n')
print(confusion_matrix(YTEST,hybridpred))
pd.DataFrame(confusion_matrix(YTEST,hybridpred),
             columns=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'],
               index=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'])


#### TRAIN A linear SVM CLASSIFIER and GET ACCURACY REPORT ###
hybrid_classifier=svm.LinearSVC()
hybrid_classifier.fit(XTRAIN,YTRAIN)

print('Hybrid Classifier Mertics - Linear SVM')
print('No. of test instances:',len(XTEST),len(YTEST))
### Accuracy Report ###
hybridpred=hybrid_classifier.predict(XTEST)
print('Overall accuracy:',accuracy_score(YTEST,hybridpred))
print(classification_report(YTEST, hybridpred, target_names=category_names))
print('Confusion matrix:\n')
print(confusion_matrix(YTEST,pred))
pd.DataFrame(confusion_matrix(YTEST,hybridpred),
             columns=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'],
               index=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'])



XTRAIN_HOG=[]
for X in XTRAIN:
    XTRAIN_HOG.append(X[daisy_cluster_model.n_clusters:])
    


XTEST_HOG=[]
for X in XTEST:
    XTEST_HOG.append(X[daisy_cluster_model.n_clusters:])
    

#### TRAIN HOG only CLASSIFIER and GET ACCURACY REPORT ###
hogclassifier=svm.LinearSVC()
hogclassifier.fit(XTRAIN_HOG,YTRAIN)

print('HOG Only Classifier Mertics')
print('No. of test instances:',len(XTEST_HOG),len(YTEST))
### Accuracy Report ###
pred=hogclassifier.predict(XTEST_HOG)
print('Overall accuracy:',accuracy_score(YTEST,pred))
print(classification_report(YTEST, pred, target_names=category_names))
print('Confusion matrix:\n')
print(confusion_matrix(YTEST,pred))
pd.DataFrame(confusion_matrix(YTEST,pred),
             columns=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'],
               index=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'])



XTRAIN_DAISY=[]
for X in XTRAIN:
    XTRAIN_DAISY.append(X[:daisy_cluster_model.n_clusters])
    


XTEST_DAISY=[]
for X in XTEST:
    XTEST_DAISY.append(X[:daisy_cluster_model.n_clusters])
    


#### TRAIN DAISY only CLASSIFIER and GET ACCURACY REPORT ###
daisyclassifier=svm.LinearSVC()
daisyclassifier.fit(XTRAIN_DAISY,YTRAIN)



print('HOG Only Classifier Mertics')
print('No. of test instances:',len(XTEST_DAISY),len(YTEST))
### Accuracy Report ###
pred = daisyclassifier.predict(XTEST_DAISY)
print('Overall accuracy:',accuracy_score(YTEST,pred))
print(classification_report(YTEST, pred, target_names=category_names))
print('Confusion matrix:\n')
print(confusion_matrix(YTEST,pred))
pd.DataFrame(confusion_matrix(YTEST,pred),
             columns=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'],
               index=['MITinsidecity', 'bedroom', 'PARoffice', 'MITmountain', 'MITtallbuilding', 'MIThighway', 'MITcoast', 'livingroom', 'MITopencountry', 'MITstreet', 'MITforest', 'kitchen', 'CALsuburb'])

