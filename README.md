# covid19-detection-chest-xray-dataset

# Overview of the Project

## Problem Statement - 
#### Analysis, Prediction and Evaluation of Covid-19 Datasets using Quanvolutional Neural Network

## Dataset used - 
We have used [this datset](https://www.kaggle.com/pranavraikokte/covid19-image-dataset) from Kaggle which contains 250 training and 65 testing images for our model. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/xray_example.jpeg?raw=true)


## Our Approach to the classifier- 

### Preprocessing the dataset
Images given in the dataset is real life chest x-ray and is not previouly modified. So all have different dimensions. So we reduced all the image size to a specific dimension. It would be more convenient to fix the image size to 256x256 to keep most of the patterns intact. But due to some limitation in computational resources, in this demonstration we kept the size as 28x28. We will definitely try with 256x256 dimension, later as an extended version of this hackathon project. 
Initially the dataset is provided in a folder format where all images of each classes are put in different folders. We used [this python script](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/script_conv_to_csv.py) to convert those images to 28x28 using openCV library and finally saved in csv format. You can get here the [train csv](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/train.csv) and [test csv](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/test.csv). 


### Applying Quanvolutional Layer
We have extended the single layer approach of [Quanvolutional Neural Network](https://pennylane.ai/qml/demos/tutorial_quanvolution.html) to multiple layers, to be exact 4 layers in our model. You can get the notebook [here](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quanvolution_on_xray_image.ipynb). 

*In case notebook doesn't render properly, you can see this [pdf](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quanvolution_on_xray_image.pdf)

Initially Each images has the dimension of (28x28x1) which is fed to the first Quanvolutional layer and converted to (14x14x4). The 2nd Layer converts it to (7x7x16), 3rd layer to (3x3x64) and finally the 4th and last layer converts each to a (1x1x256) dimensional data matrix. 
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/qnn.png?raw=true)


Here although the parameters of the Quanvolutional Layer gates are uniformly randomized, We have considered the approach of training these parameters too and see if there is any improvement in the result later in the extended version of this project. 

### Classifier Model
After the Quanvolutional layers, we have the classifier model. The classifier model consists of two subclassifiers each of which is a binary classifier. We dente those two by 'Model-1' and 'Model-2'. 

Model-1 classifies between two classes - 'Normal Person' and 'Covid19/Viral Pnemonia'. 

Model-2 classifies between two classes - 'Covid-19' and 'Viral Pneumonia'. 

One important point to mention here, Intially we tried to use single model for multi class classification and result turned out to be quiet low. So the two model approach we considered as the final one. 

Also Here the accuracy of Model-1 is always higher than model-2 because differentiating 'Noraml Person' vs "covid/Viral Pneumonia' is easier than differentiating between 'covid' vs 'Viral Pneumonia'. 

![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/models.png)

##### We have created two notebooks for this. 

In [Notebook-1](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quantum_classifier_1.ipynb) we have used 11 features from 256 feature sized input data. 
In [Notebook-2](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/quantum_classifier_2.ipynb) we have reduced 256 features of each image to 4 using TruncatedSVD method. 

We have done this beacuse encoding 256 features to a quantum circuit is not a feasible approach. 

### Prediction
While Predicting, we first give input to the Model-1. If it predicts as Normal person, then it is the final prediction assigned to the input. If not, then we give the same input to Model-2 and it finally predicts whether the chest x-ray reveals the patients has Covid-19 or Viral Pneumonia.

### Plots for Training cost and accuracy for Model-1 and Model-2

#### Cost Plot for Model-1
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/cost_plot_model_1.png?raw=true)
#### Training accuracy plot for Model-1
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/train_acc_plot_model_1.png?raw=true)
#### Cost Plot for Model-2
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/cost_plot_model_2.png?raw=true)
#### Training accuracy plot for Model-2
![alt text](https://github.com/QTechnocrats/covid19-detection-chest-xray-dataset/blob/main/images/train_acc_plot_model_2.png?raw=true)


# Some drawbacks in the draft due to lack of computational resources - 

1. The real x-ray images in the dataset is enough large to contain a lots of information. But due to lack of computational resources we reduced the size to 28x28 using openCV library, which may have suppressed a lot of important informations. 
Later with the availability of more computational resources we can use 256x256 dimensional image which will incrase the accuracy of the model. 

2. Currently although after applying Quanvolution and flattening the data we had 256 features of each image, we used only 11 features by feature selection method due to lack of resources. We can try out with more features with availability of computational resources to experiment how accuracy imporves with more number of features. 

3. Currently all the training and prediction has been done on simulator, not real Quantum computer (as it has been recommended not to train circuit on QC with that 250 USD credit), but we are really hopeful about some improvements if training is executed in a real Quantum computer, not a simulator.

Now while training each model, number of times we call quantum circuit is equal to
 (shots assigned to the device * number of training images * number of iteration)
with this above rough calculation and as we have 250 training images, the resources required can be predicted easily.

4. All the four quanvolutional layers applied on the image data is using uniformly generated random parameters, which are not furthet trained. 

Later, we can experiment with training those quanvolutional layers too the way its done in classical convolution.
