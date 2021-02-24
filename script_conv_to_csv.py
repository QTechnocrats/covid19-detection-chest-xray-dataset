from os import scandir, getcwd
import cv2
import numpy as np
import pandas as pd

def ls(ruta):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

width = 28
height = 28

####### test #############
test_covid_path = 'test/Covid/'
test_normal_path = 'test/Normal/'
test_pneumonia_path = 'test/Viral Pneumonia/'
list_covid_test = []
list_normal_test = []
list_pneumonia_test = []

#print('Covid')

files_0 = ls(test_covid_path)

for image in files_0:
	img = cv2.imread(test_covid_path+image, 0) 
	dim = (width, height) 
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),0)
	list_covid_test.append(imagen_entrada)

df_covid_test = pd.DataFrame(data=list_covid_test, index=[i for i in range(len(list_covid_test))])
#print(df_covid_test)



#print('Normal')
files_1 = ls(test_normal_path)
for image in files_1:
	img = cv2.imread(test_normal_path+image, 0) 
	dim = (width, height) 
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),1)
	list_normal_test.append(imagen_entrada)

df_normal_test = pd.DataFrame(data=list_normal_test, index=[i for i in range(len(list_normal_test))])
#print(df_normal_test)


#print('Pneumonia')
files_2 = ls(test_pneumonia_path)
for image in files_2:
	img = cv2.imread(test_pneumonia_path+image, 0) 
	dim = (width, height) 
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),2)
	list_pneumonia_test.append(imagen_entrada)

df_pneumonia_test = pd.DataFrame(data=list_pneumonia_test, index=[i for i in range(len(list_pneumonia_test))])
#print(df_pneumonia_test)


dframes_test = [df_covid_test, df_normal_test, df_pneumonia_test]
df_test = pd.concat(dframes_test)
print(df_test)
df_test.to_csv('test.csv',header=None,index=False)



########## train ###########

train_covid_path = 'train/Covid/'
train_normal_path = 'train/Normal/'
train_pneumonia_path = 'train/Viral Pneumonia/'
list_covid_train = []
list_normal_train = []
list_pneumonia_train = []



#print('Covid')

files_0 = ls(train_covid_path)

for image in files_0:
	img = cv2.imread(train_covid_path+image, 0) 
	dim = (width, height) 
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),0)
	list_covid_train.append(imagen_entrada)

df_covid_train = pd.DataFrame(data=list_covid_train, index=[i for i in range(len(list_covid_train))])
#print(df_covid_train)



#print('Normal')
files_1 = ls(train_normal_path)
for image in files_1:
	img = cv2.imread(train_normal_path+image, 0) 
	dim = (width, height) 
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),1)
	list_normal_train.append(imagen_entrada)

df_normal_train = pd.DataFrame(data=list_normal_train, index=[i for i in range(len(list_normal_train))])
#print(df_normal_train)


#print('Pneumonia')
files_2 = ls(train_pneumonia_path)
for image in files_2:
	img = cv2.imread(train_pneumonia_path+image, 0) 
	dim = (width, height) 
	imagen_entrada = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	imagen_entrada = imagen_entrada.flatten()
	imagen_entrada = np.insert(imagen_entrada,len(imagen_entrada),2)
	list_pneumonia_train.append(imagen_entrada)

df_pneumonia_train = pd.DataFrame(data=list_pneumonia_train, index=[i for i in range(len(list_pneumonia_train))])
#print(df_pneumonia_train)


dframes_train = [df_covid_train, df_normal_train, df_pneumonia_train]
df_train = pd.concat(dframes_train)
print(df_train)
df_train.to_csv('train.csv',header=None,index=False)
