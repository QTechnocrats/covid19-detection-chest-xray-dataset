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
We have extended the single layer approach of Quanvolutional Neural Network from [here](https://pennylane.ai/qml/demos/tutorial_quanvolution.html) to multiple layers, to be exact 4 layers in our model. You can get the notebook [here](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quanvolution_on_xray_image.ipynb). 

*In case notebook doesn't render properly, you can see this [pdf](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quanvolution_on_xray_image.pdf)

Initially Each images has the dimension of (28x28x1) which is fed to the first Quanvolutional layer and converted to (14x14x4). The 2nd Layer converts it to (7x7x16), 3rd layer to (3x3x64) and finally the 4th and last layer converts each to a (1x1x256) dimensional data matrix. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/qnn.png?raw=true)

### Classifier Model
After the Quanvolutional layers, we have the classifier model. The classifier model consists of two subclassifiers each of which is a binary classifier. We dente those two by 'Model-1' and 'Model-2'.

Model-1 classifies between two classes - 'Normal Person' and 'Covid19/Viral Pnemonia'. 

Model-2 classifies between two classes - 'Covid10' and 'Viral Pneumonia'. 

##### We have created two notebooks for this. 

In [Notebook-1](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quantum_classifier_1.ipynb) we have used 11 features from 256 feature sized input data. 
In [Notebook-2](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quantum_classifier_2.ipynb) we have reduced 256 features of each image to 4 using TruncatedSVD method. 

We have done this beacuse encoding 256 features to a quantum circuit is not a feasible approach. 

### Prediction
While Predicting, we first give input to the Model-1. If it predicts as Normal person, then it is the final prediction assigned to the input. If not, then we give the same input to Model-2 and it finally predicts whether the chest x-ray is Covid10 patient or Viral Pneumonia patient.

### Plots for Trainging cost and accuracy for Model-1 and Model-2

#### Cost Plot for Model-1
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/cost_plot_model_1.png?raw=true)
#### Trainging accuracy plot for Model-1
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/train_acc_plot_model_1.png?raw=true)
#### Cost Plot for Model-2
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/cost_plot_model_2.png?raw=true)
#### Trainging accuracy plot for Model-2
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/train_acc_plot_model_2.png?raw=true)
