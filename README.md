# Deep Learning Models on different scenarios

These Notebooks with their Question Statement and Reports, came under the course CSL7590, taken by Prof. Mayank Vatsa.

## Lab-1 (NN from Scratch, Denoising Autoencoder and CNN)

Question-1:

Implement back propagation neural network from scratch on Letter Recognition dataset with following configurations:-

a.) Vary learning rate

b.) Vary number of epochs

c.) Xavier weight initialisation

d.) Use Adam optimizer

e.) Using activation functions: tanh, ReLU

Use cross-entropy loss. Plot the loss and accuracy curves on the training and test sets. Find the best configuration for your network (a combination of best learning rate, best number of epochs, and activation function), support your claim.

Question-2:

Implement a Denoising Autoencoder using three encoding and three decoding layers on the MNIST dataset for construction and reconstruction of the image. Use MSE as the loss function:-

a.) Use 1 FC layer and 2 different activation functions of your choice for 10-class classification

b.) Use 3 FC layers and 2 different activation functions of your choice for 10-class classification

c.) Compare the performance between 1 FC and 3 FC layer results and report the accuracy on test set and plot loss curves on training and test dataset

Question-3:

Construct following CNN architectures. Use the CIFAR-10 dataset for all the analysis:-

1.) Conv-Conv-Pool-Conv-Conv-Pool-FC-FC

2.) Conv-Pool-BatchNormalization-ReLU-Conv-Pool-BatchNormalization-ReLU-FC

3.) Conv-BatchNorm-ReLU-Conv-BatchNorm-ReLU-FC

FC refers to Fully Connected Layer here, provide with the following analysis:-

a.) How does changing network size affect the performance ?

b.) Experiment with different sizes and types of pooling like (Max Pooling, Average Pooling and Stochastic Pooling(optional)) and do a detailed analysis of pooling size on the network.

c.) How the presence of one or more fully connected layers changes the accuracy.

d.) Change the stride size from 1 to 2 and find the differences in the output size.

## Lab-2 (DenseNet Fine Tuning, ResNet+SVM and CNN)

Question-1:

Download a ResNet 50 trained on the ImageNet classification dataset:-

a.) Use the features extracted from the last fully-connected layer and train a multiclass SVM
classifier on STL-10 dataset. Report the following:- 
i.) Accuracy, Confusion Matrix on test data. 
ii.) ROC curve (assuming the chosen class as positive class and remaining classes as negative) 

b.) Fine-tune the ResNet 50 model (you may choose what layers to fine-tune) for the STL-10 dataset, and evaluate the classification performance on the test set before and after fine-tuning with respect to the following metrics:-
i.) Class wise Accuracy 
ii.) Report Confusion Matrix. 

[Code for accuracy, ROC, Confusion Matrix should be done from scratch, SVM - you may use sklearn]

Question-2:

Download the Tiny ImageNet dataset from here and pick the first 50 classes from 200 of them. Finetune ResNet-18 with the following:-

a.) Cross-Entropy as the final classification loss function 

b.) Center loss as the final classification loss function 

c.) Triplet Loss as the final classification loss function 

Choose any evaluation metrics (at least 3) and compare the models in a, b and c, comment on which one is better and why?

Question-3:

Implement a three-layer CNN network, for classification tasks for the Dogs vs. Cats dataset. [You can use the necessary libraries/modules]

a.) Compare the accuracy on the test dataset (split into train and test [70:30]) for the following optimization techniques:-
i.) Vanilla SGD
ii.) Mini Batch SGD
iii.) Mini Batch with momentum
iv.) Mini Batch with Adam

b.) What are your preferred mini-batch sizes? Explain your choice with proper gradient update plots.

c.) What are the advantages of shuffling and partitioning the mini-batches? 

## Lab-3 and 4 (DCGANs, Few-Shot Learning and Darts)

Question-1:

Train a DCGAN to generate images from noise. Use the MNIST database to learn the GAN Network.

Discriminator in DCGAN:-

i.) if roll no. % 2 == 0: use VGG16 as a discriminator.
ii.) if roll no. % 2 == 1: use Resnet-18 as a discriminator.]

Perform the following tasks:-

a.) Uniformly generate ten noise vectors that act as latent representation vectors, and generate the images for these noise vectors, and visualize them at:-
i.) After the first epoch.
ii.) After n/2 th epoch.
iii.) After your last epoch. (say n epochs in total) and comment on the image interpretation at (i), (ii) and (iii) and can you identify the images? 

b.) Plot generator and discriminator losses for all the iterations. [One iteration = forward pass of a mini-batch]

c.) Do we have control of what class image the generator will generate, given a noise vector? Suppose, we are interested in generating only “4” images, will the GAN you trained in (a) can do that? Explain why. If not, modify the GAN trained above to do this. 

Question-2:

Reproduce the results of the paper “Prototypical Networks for Few-shot Learning”. You can use the authors’ code (mostly provided via GitHub) or any other reimplementation available on the internet. [Please cite the source].

You should be able to understand and explain the code that you are using. Write a 2 page report explaining the algorithm and your analysis based on your results. 

Question-3:

Your task is to train a GAN that generates maps given satellite images. 

a.) Report visualized image translations [satellite image -ground truth map - map generated]. [at least 10 examples] 

b.) Plot generator and discriminator losses for all the iterations. [One iteration = forward pass of a mini-batch]

c.) Use the pre-trained sat2map generator modeland generate images for the same ten examples in (a). Compare and comment on the images generated with a pre-trained generator with your trained generator in (a). Support your claim with SSIM. 

Question-4:

Reproduce the results of the paper “DARTS: Differentiable Architecture Search” only on CIFAR-10 dataset. You can use the authors’ code (mostly provided via GitHub) or any other reimplementation available on the internet. [Please cite the source].

You should be able to understand and explain the code that you are using. Write a 2 page report explaining the algorithm and your analysis based on your results. 
