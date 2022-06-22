# Deep Learning Models on different scenarios

These Notebooks with their Question Statement and Reports, came under the course CSL7590, taken by Prof. Mayank Vatsa.

## Lab-1 (NN from Scratch, Denoising Autoencoder and CNN)

Question-1:

![0__SH7tsNDTkGXWtZb](https://user-images.githubusercontent.com/54277039/175081252-51d62fa1-fad4-4b68-ab28-9b2e9a6bb81b.png)

Implement back propagation neural network from scratch on Letter Recognition dataset with following configurations:-

a.) Vary learning rate

b.) Vary number of epochs

c.) Xavier weight initialisation

d.) Use Adam optimizer

e.) Using activation functions: tanh, ReLU

Use cross-entropy loss. Plot the loss and accuracy curves on the training and test sets. Find the best configuration for your network (a combination of best learning rate, best number of epochs, and activation function), support your claim.

Question-2:

![0_qgxeODdkcdMIfR--](https://user-images.githubusercontent.com/54277039/175081589-3dae5b5a-7f0b-4074-8c10-855023d4c1be.png)

Implement a Denoising Autoencoder using three encoding and three decoding layers on the MNIST dataset for construction and reconstruction of the image. Use MSE as the loss function:-

a.) Use 1 FC layer and 2 different activation functions of your choice for 10-class classification

b.) Use 3 FC layers and 2 different activation functions of your choice for 10-class classification

c.) Compare the performance between 1 FC and 3 FC layer results and report the accuracy on test set and plot loss curves on training and test dataset

Question-3:

![1_uAeANQIOQPqWZnnuH-VEyw](https://user-images.githubusercontent.com/54277039/175081765-4ce3606e-d540-4337-9916-faaae135afe0.jpg)

Construct following CNN architectures. Use the CIFAR-10 dataset for all the analysis:-

1.) Conv-Conv-Pool-Conv-Conv-Pool-FC-FC

2.) Conv-Pool-BatchNormalization-ReLU-Conv-Pool-BatchNormalization-ReLU-FC

3.) Conv-BatchNorm-ReLU-Conv-BatchNorm-ReLU-FC

FC refers to Fully Connected Layer here, provide with the following analysis:-

a.) How does changing network size affect the performance ?

b.) Experiment with different sizes and types of pooling like (Max Pooling, Average Pooling and Stochastic Pooling(optional)) and do a detailed analysis of pooling size on the network.

c.) How the presence of one or more fully connected layers changes the accuracy.

d.) Change the stride size from 1 to 2 and find the differences in the output size.

### References

For Q-1 :-

● https://github.com/jiexunsee/Adam-Optimizer-from-scratch/blob/master/adamoptimizer.py

● https://towardsdatascience.com/the-ultimate-beginners-guide-to-implement-a-neural-network-from-scratch-cf7d52d91e00

For Q-2 :-

● https://www.geeksforgeeks.org/implementing-an-autoencoder-inpytorch/

● https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e

● https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a

● https://nextjournal.com/gkoehler/pytorch-mnist

For Q-3 :-

● https://shonit2096.medium.com/cnn-on-cifar10-data-set-using-pytorch-34be87e09844

● https://pytorch.org/docs/stable/index.html

## Lab-2 (DenseNet Fine Tuning, ResNet+SVM and CNN)

Question-1:

Download a ResNet 50 trained on the ImageNet classification dataset:-

![ResNet-50-convolutional-neural-networks-with-SVM-ResNet-50-convolutional-neural](https://user-images.githubusercontent.com/54277039/175081989-35efebc7-255d-489d-a43d-1fa57939098d.png)

a.) Use the features extracted from the last fully-connected layer and train a multiclass SVM
classifier on STL-10 dataset. Report the following:- 
i.) Accuracy, Confusion Matrix on test data. 
ii.) ROC curve (assuming the chosen class as positive class and remaining classes as negative) 

b.) Fine-tune the ResNet 50 model (you may choose what layers to fine-tune) for the STL-10 dataset, and evaluate the classification performance on the test set before and after fine-tuning with respect to the following metrics:-
i.) Class wise Accuracy 
ii.) Report Confusion Matrix. 

[Code for accuracy, ROC, Confusion Matrix should be done from scratch, SVM - you may use sklearn]

Question-2:

![Tiny_ImageNet-0000001404-a53923c3_XCrVSGm](https://user-images.githubusercontent.com/54277039/175083057-36b54b51-b7c3-4b05-a01a-b055c7194df9.jpg)

Download the Tiny ImageNet dataset from here and pick the first 50 classes from 200 of them. Finetune ResNet-18 with the following:-

a.) Cross-Entropy as the final classification loss function 

b.) Center loss as the final classification loss function 

c.) Triplet Loss as the final classification loss function 

Choose any evaluation metrics (at least 3) and compare the models in a, b and c, comment on which one is better and why?

Question-3:

![1_-_2dKCQHh_-_Long_Valley](https://user-images.githubusercontent.com/54277039/175083627-5849e392-4972-4655-987c-84a32e0b03f2.gif)

Implement a three-layer CNN network, for classification tasks for the Dogs vs. Cats dataset. [You can use the necessary libraries/modules]

a.) Compare the accuracy on the test dataset (split into train and test [70:30]) for the following optimization techniques:-
i.) Vanilla SGD
ii.) Mini Batch SGD
iii.) Mini Batch with momentum
iv.) Mini Batch with Adam

b.) What are your preferred mini-batch sizes? Explain your choice with proper gradient update plots.

c.) What are the advantages of shuffling and partitioning the mini-batches? 

### References

Question-1 :-

https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6#:~:text=In%20PyTorch%20we%20can%20freeze,to%20apply%20a%20pretrained%20model.

https://gist.github.com/L0SG/2f6d81e4ad119c4f798ab81fa8d62d3f

https://blog.ineuron.ai/AUC-ROC-score-and-curve-in-multiclass-classification-problems-2ja4jOHb2X

https://github.com/pytorch/vision/issues/48

StackOverflow and Pytorch Documentations

Question-2 :-

https://github.com/KaiyangZhou/pytorch-center-loss

https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b

StackOverflow Documentations

Question-3 :-

https://github.com/amitrajitbose/cat-v-dog-classifier-pytorch/blob/master/train.py

https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

StackOverflow and Geeks-For-Geeks Documentations

## Lab-3 and 4 (DCGANs, Few-Shot Learning and Darts)

Question-1:

![download](https://user-images.githubusercontent.com/54277039/175083875-83a5603d-9035-460d-aaa4-4d6c54b83c56.png)

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

![download](https://user-images.githubusercontent.com/54277039/175084045-7aa07dc3-b786-4ce7-a976-cbb55ea27488.jpg)

Reproduce the results of the paper “Prototypical Networks for Few-shot Learning”. You can use the authors’ code (mostly provided via GitHub) or any other reimplementation available on the internet. [Please cite the source].

You should be able to understand and explain the code that you are using. Write a 2 page report explaining the algorithm and your analysis based on your results. 

Question-3:

![42405_2021_396_Fig1_HTML](https://user-images.githubusercontent.com/54277039/175084229-b55f370b-7a9a-46f6-b680-295d4956aed4.png)

Your task is to train a GAN that generates maps given satellite images. 

a.) Report visualized image translations [satellite image -ground truth map - map generated]. [at least 10 examples] 

b.) Plot generator and discriminator losses for all the iterations. [One iteration = forward pass of a mini-batch]

c.) Use the pre-trained sat2map generator modeland generate images for the same ten examples in (a). Compare and comment on the images generated with a pre-trained generator with your trained generator in (a). Support your claim with SSIM. 

Question-4:

![download](https://user-images.githubusercontent.com/54277039/175084515-88268860-005c-43fa-9bd1-843273e3e977.png)

Reproduce the results of the paper “DARTS: Differentiable Architecture Search” only on CIFAR-10 dataset. You can use the authors’ code (mostly provided via GitHub) or any other reimplementation available on the internet. [Please cite the source].

You should be able to understand and explain the code that you are using. Write a 2 page report explaining the algorithm and your analysis based on your results. 

