# Melanoma Detection
To develop a convolutional neural network (CNN) to detect melanoma and other skin diseases from images. Melanoma is a deadly skin cancer requiring early detection. Using the ISIC dataset having images, classify skin conditions into 9 categories. To built a CNN model to accurately identify the skin conditions to aid dermatologists in early diagnosis and treatment.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
The dataset includes images of skin conditions, both cancerous and non-cancerous, from the International Skin Imaging Collaboration (ISIC). These images were grouped based on ISIC's classification. To ensure equal representation, the Augmentor tool was used to create additional images for underrepresented classes. This prevents class imbalance. The goal is to identify and label specific types of skin cancer from these images.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
The CNN architecture begins with data preparation, artificially expanding the training dataset through random transformations (rotation, scaling, flipping) using the augmentation data variable. Input image pixel values are then scaled to 0-1 range using Rescaling(1./255) for stable training.

Three sequential convolutional layers are applied using Conv2D, each with ReLU activation for non-linearity, padding='same' to maintain spatial dimensions, and an increasing number of filters (16, 32, 64) for deeper feature maps. Max-pooling layers (MaxPooling2D) follow each convolutional layer to downsample and retain key information.

A dropout layer (Dropout) with a 20% rate is applied to prevent overfitting, and a flatten layer transforms 2D feature maps into 1D vectors. Two dense layers (Dense) with ReLU activation are then used: the first with 128 neurons, and the second outputting final classification probabilities.

The output layer's neuron count is determined by the target_labels variable (number of classes), with no activation function specified (as it's followed by the loss function during training).

The model is compiled using the Adam optimize and Sparse Categorical Crossentropy loss function, with accuracy as the evaluation metric (metrics=['accuracy']).

Training occurs using the fit method, with ModelCheckpoint and EarlyStopping callbacks monitoring validation accuracy. ModelCheckpoint saves the best validation accuracy model, while EarlyStopping stops training if validation accuracy stops changing much.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
Matplotlib
Numpy
Pandas
Seaborn
Tensorflow

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
UpGrad/IIITB


## Contact
Created by [@kudumulaanuja]


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->