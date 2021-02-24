# covid19-detection-chest-xray-dataset

# Overview of the Project

## Problem Statement - 
#### Detecting Covid19 from Chest X-ray images of patient using Quantum Circuit

## Dataset used - 
We have used [this datset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset) from Kaggle which contains 250 training and 65 testing images for our model. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/xray_example.jpeg?raw=true)


## Our Approach to the classifier- 

### Preprocessing the dataset
Images given in the dataset is real life chest x-ray and is not previouly modified. So all have different dimensions. So we reduced the all the image size to a specific dimension. It would be more convenient to fix the image size to 256x256 but due to limitations of computational resources we have to reduce it to 28x28 size. 
Initially the dataset is provided in a folder format where all images of each classes are put in different folders. We used [this python script](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/script_conv_to_csv.py) to convert those images to 28x28 using openCV library and finally saved in csv format. You can get here the [train csv](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/train.csv) and [test csv](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/test.csv). 


### Applying Quanvolutional Layer
We have extended the single layer approach of Quanvolutional Neural Network from [here]() to multiple layers, to be exact 4 layers in our model. 
Initially Each images has the dimension of (28x28x1) which is fed to the first Quanvolutional layer and converted to (14x14x4). The 2nd Layer converts it to (7x7x16), 3rd layer to (3x3x64) and finally the 4th and last layer converts each to a (1x1x256) dimensional data matrix. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/qnn.png?raw=true)

