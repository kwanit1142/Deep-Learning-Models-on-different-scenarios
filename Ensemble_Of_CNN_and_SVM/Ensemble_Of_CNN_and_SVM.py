# -*- coding: utf-8 -*-
"""B19EE046_DL_Assignment_2_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LEnGeb_u8ooCUZOg6CqGk9ul1NLfmM2z
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Importing Libraries"""

import numpy as np
import os
import torch
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from random import shuffle
import pandas as pd
from torch.utils.data import Dataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""# Dataset Handling

## Fetching Dataset from Web-link
"""

!wget 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz' -P '/content/drive/MyDrive/DL_Assignment_2/'

"""## Unzipping the tar.gz file"""

!tar -xvf  '/content/drive/MyDrive/DL_Assignment_2/stl10_binary.tar.gz' -C '/content/drive/MyDrive/DL_Assignment_2/Q_1'

"""## Utility Functions to open bin files and construct meta-file (image-name and labels)"""

def bin_to_file(path):
  with open(path, 'rb') as f:
    file_open = np.fromfile(f,dtype=np.uint8)
    return file_open

def save_image_and_meta(directory, images_array, label_array):
  df = pd.DataFrame(columns=['Image_Name','Image_Class'])
  img_name_list=[]
  img_class_list=[]
  idx=0
  for image in tqdm(images_array):
    image = image.transpose((2,1,0))
    label = label_array[idx]
    img_name = 'Img_'+str(idx)+'.png'
    file_directory = directory+'/'+img_name
    img_name_list.append(img_name)
    img_class_list.append(label)
    idx = idx+1
    cv2.imwrite(file_directory,image)
  df['Image_Name']=img_name_list
  df['Image_Class']=img_class_list
  save_directory = directory+'meta.csv'
  df.to_csv(save_directory)

"""### X_Unsupervised"""

x_u = bin_to_file('/content/drive/MyDrive/DL_Assignment_2/Q_1/stl10_binary/unlabeled_X.bin')
x_u = x_u.reshape((-1, 3, 96, 96))
x_u.shape

"""### X_Train"""

x_train = bin_to_file('/content/drive/MyDrive/DL_Assignment_2/Q_1/stl10_binary/train_X.bin')
x_train = x_train.reshape((-1, 3, 96, 96))
x_train.shape

"""### X_Test"""

x_test = bin_to_file('/content/drive/MyDrive/DL_Assignment_2/Q_1/stl10_binary/test_X.bin')
x_test = x_test.reshape((-1, 3, 96, 96))
x_test.shape

"""### Y_Train"""

y_train = bin_to_file('/content/drive/MyDrive/DL_Assignment_2/Q_1/stl10_binary/train_y.bin')
y_train.shape

"""### Y_Test"""

y_test = bin_to_file('/content/drive/MyDrive/DL_Assignment_2/Q_1/stl10_binary/test_y.bin')
y_test.shape

save_image_and_meta('/content/drive/MyDrive/DL_Assignment_2/Q_1/train/',x_train,y_train)
save_image_and_meta('/content/drive/MyDrive/DL_Assignment_2/Q_1/test/',x_test,y_test)

"""# Converting Dataset into Torch-Dataset, via Class Preparation and DataLoader"""

class Data_Prepare(Dataset):
  """
  The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
  """
  def __init__(self, data_frame, root_dir, transform=None):
    self.data_frame = data_frame
    self.root_dir = root_dir
    self.transform = transform
  def __len__(self):
    # Return the length of the dataset
    return len(self.data_frame)
  def __getitem__(self, idx):
    # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
    if torch.is_tensor(idx):
      idx = idx.tolist()    
    img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
    image = Image.open(img_name)
    label = (self.data_frame.iloc[idx, -1])-1
    if self.transform:
      image = self.transform(image)
    return (image, label)

def data_preparation(Data_Class, root_directory_train, root_directory_test, train_df, test_df, Mean, Std, Batch_Size = 128, Shuffle = False):
  transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(Mean, Std)])
  train_dataset = Data_Class(data_frame=train_df,root_dir=root_directory_train,transform = transform)
  train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Batch_Size, shuffle = Shuffle, num_workers=2, pin_memory=True)
  test_dataset = Data_Class(data_frame=test_df,root_dir=root_directory_test,transform = transform)
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Batch_Size, shuffle = Shuffle, num_workers=2, pin_memory=True)
  return train_loader, test_loader

train_df = pd.read_csv('/content/drive/MyDrive/DL_Assignment_2/Q_1/train/meta.csv')
test_df = pd.read_csv('/content/drive/MyDrive/DL_Assignment_2/Q_1/test/meta.csv')
train_df = train_df.drop(columns=['Unnamed: 0'])
test_df = test_df.drop(columns=['Unnamed: 0'])

train_df

train,test = data_preparation(Data_Prepare, '/content/drive/MyDrive/DL_Assignment_2/Q_1/train', '/content/drive/MyDrive/DL_Assignment_2/Q_1/test', train_df, test_df, (0,0,0),(1,1,1), Batch_Size = 1024, Shuffle=True )

"""# Importing Pre-Trained ResNet-50

## Forward-passing Train and Test Images
"""

resnet_model = models.resnet50(pretrained=True).cuda()
for param in resnet_model.parameters():
  param.requires_grad = False

x_train = []
x_test = []

for (image, _) in train:
    image = image.cuda()
    out = resnet_model.forward(image)
    x_train.append(out)

for (image, _) in test:
    image = image.cuda()
    out = resnet_model.forward(image)
    x_test.append(out)

"""## Converting Labels into CUDA-Labels"""

label_train = []
label_test = []

for (_, label) in train:
    label_train.append(label.cuda())

for (_, label) in test:
    label_test.append(label.cuda())

"""# Converting Torch Matrix into Feature-Array

## X_Train
"""

x_train_array = x_train[0].cpu().numpy()
for i in x_train[1:]:
  x_train_array = np.concatenate((x_train_array,i.cpu().numpy()),axis=0)
x_train_array.shape

"""## X_Test"""

x_test_array = x_test[0].cpu().numpy()
for i in x_test[1:]:
  x_test_array = np.concatenate((x_test_array,i.cpu().numpy()),axis=0)
x_test_array.shape

"""## Y_Train"""

label_train_array = label_train[0].cpu().numpy()
for i in label_train[1:]:
  label_train_array = np.concatenate((label_train_array,i.cpu().numpy()),axis=0)
label_train_array.shape

"""## Y_Test"""

label_test_array = label_test[0].cpu().numpy()
for i in label_test[1:]:
  label_test_array = np.concatenate((label_test_array,i.cpu().numpy()),axis=0)
label_test_array.shape

"""# Normalization Methods for variety of Distribution-related Trends"""

ss = StandardScaler()
mms = MinMaxScaler()
ss.fit(x_train_array)
mms.fit(x_train_array)
x_train_array_ss = ss.transform(x_train_array)
x_train_array_mms = mms.transform(x_train_array)
x_test_array_ss = ss.transform(x_test_array)
x_test_array_mms = mms.transform(x_test_array)

"""# Confusion-Matrix and Overall Accuracy"""

def first_subpart(pred, true):
  length = true.max()
  confusion_matrix = [[0 for i in range(length)] for j in range(length)]
  for row in range(0,len(true)):
    column = pred[row]
    confusion_matrix[true[row]-1][pred[row]-1] +=1
  print("Confusion Matrix\n")
  for i in confusion_matrix:
    print(i)
  print("\nOverall-Accuracy\n")
  correct=0
  for i in range(length):
    correct += confusion_matrix[i][i]
  print(str(correct*100/len(true))+"%")

"""## ROC-Curve for One-v/s-Rest Configuration"""

def second_subpart(model_class, test_array, test_label):
  fpr = {}
  tpr = {}
  thresh ={}
  n_class = test_label.max()
  plt.figure(figsize=(10,10))
  for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(test_label, model_class.predict_proba(test_array)[:,i], pos_label=i)
    plt.plot(fpr[i], tpr[i], label='Class '+str(i)+' vs Rest')
  plt.title('Multiclass ROC curve')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive rate')
  plt.legend(loc='best')
  plt.show()

"""## RBF Kernel with Normalization Methods"""

rbf = SVC(probability=True)
rbf.fit(x_train_array,label_train_array)
pred_rbf = rbf.predict(x_test_array)

rbf_ss = SVC(probability=True)
rbf_ss.fit(x_train_array_ss,label_train_array)
pred_rbf_ss = rbf_ss.predict(x_test_array_ss)

rbf_mms = SVC(probability=True)
rbf_mms.fit(x_train_array_mms,label_train_array)
pred_rbf_mms = rbf_mms.predict(x_test_array_mms)

first_subpart(pred_rbf,label_test_array)

first_subpart(pred_rbf_ss,label_test_array)

first_subpart(pred_rbf_mms,label_test_array)

second_subpart(rbf, x_test_array, label_test_array)

second_subpart(rbf_ss, x_test_array_ss, label_test_array)

second_subpart(rbf_mms, x_test_array_mms, label_test_array)

"""## Linear Kernel with Normalization Methods"""

linear = SVC(kernel='linear', probability=True)
linear.fit(x_train_array,label_train_array)
pred_linear = linear.predict(x_test_array)

linear_ss = SVC(kernel='linear', probability=True)
linear_ss.fit(x_train_array_ss,label_train_array)
pred_linear_ss = linear_ss.predict(x_test_array_ss)

linear_mms = SVC(kernel='linear', probability=True)
linear_mms.fit(x_train_array_mms,label_train_array)
pred_linear_mms = linear_mms.predict(x_test_array_mms)

first_subpart(pred_linear,label_test_array)

first_subpart(pred_linear_ss,label_test_array)

first_subpart(pred_linear_mms,label_test_array)

second_subpart(linear, x_test_array, label_test_array)

second_subpart(linear_ss, x_test_array_ss, label_test_array)

second_subpart(linear_mms, x_test_array_mms, label_test_array)

"""## 3rd Degree Polynomial (Cubic) Kernel with Normalization Methods"""

poly = SVC(kernel='poly', degree=3, probability=True)
poly.fit(x_train_array,label_train_array)
pred_poly = poly.predict(x_test_array)

poly_ss = SVC(kernel='poly', degree=3, probability=True)
poly_ss.fit(x_train_array_ss,label_train_array)
pred_poly_ss = poly_ss.predict(x_test_array_ss)

poly_mms = SVC(kernel='poly', degree=3, probability=True)
poly_mms.fit(x_train_array_mms,label_train_array)
pred_poly_mms = poly_mms.predict(x_test_array_mms)

first_subpart(pred_poly,label_test_array)

first_subpart(pred_poly_ss,label_test_array)

first_subpart(pred_poly_mms,label_test_array)

second_subpart(poly, x_test_array, label_test_array)

second_subpart(poly_ss, x_test_array_ss, label_test_array)

second_subpart(poly_mms, x_test_array_mms, label_test_array)

"""# Utility Functions to Train a Pytorch Model"""

def grad_change(Loss_Function, Optimizer, Label = None, Predicted = None):
  Optimizer.zero_grad()
  loss = Loss_Function(Predicted, Label)
  loss.backward()
  Optimizer.step()
  return loss, Optimizer

def model(Train_Loader, Test_Loader, Epochs, Model_Class=None, Loss_Function=None, Optimizer=None):
  outputs_train=[]
  outputs_test=[]
  for Epoch in range(Epochs):
    running_loss_train=0
    running_loss_test=0
    correct_train=0
    correct_test=0
    for (image, label) in Train_Loader:
      image = image.cuda()
      label = label.cuda()
      out = Model_Class(image)
      loss, Optimizer = grad_change(Loss_Function = Loss_Function, Optimizer = Optimizer, Label = label, Predicted = out)
      running_loss_train += loss.item()
      predicted_train = out.data.max(1, keepdim=True)[1]
      correct_train += predicted_train.eq(label.data.view_as(predicted_train)).sum()
    outputs_train.append((Epoch, running_loss_train/len(Train_Loader.dataset), 100*correct_train/len(Train_Loader.dataset)))
    with torch.no_grad():
      for (image, label) in Test_Loader:
        image = image.cuda()
        label = label.cuda()
        out = Model_Class(image)
        loss = Loss_Function(out,label)
        running_loss_test += loss.item()
        predicted_test = out.data.max(1, keepdim=True)[1]
        correct_test += predicted_test.eq(label.data.view_as(predicted_test)).sum()
      outputs_test.append((Epoch, running_loss_test/len(Test_Loader.dataset), 100*correct_test/len(Test_Loader.dataset)))
  return Model_Class, outputs_train, outputs_test

"""# Default ResNet Configuration"""

resnet_model_default = models.resnet50(pretrained=True)
for param in resnet_model_default.parameters():
    param.requires_grad = False

resnet_model_default.fc = torch.nn.Linear(2048,10)
resnet_model_default.cuda()

loss_function = torch.nn.CrossEntropyLoss().cuda()
optimizer_default = torch.optim.Adam(resnet_model_default.fc.parameters())
resnet_model_default,output_train_default,output_test_default = model(train,test,15,resnet_model_default,loss_function,optimizer_default)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_default[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_default[i][1] for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss v/s Epochs")
plt.legend(["Train","Test"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_default[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_default[i][2].cpu().numpy() for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs")
plt.legend(["Train","Test"])
plt.show()

"""# Bifurcation of ResNet Layers"""

resnet_model = models.resnet50(pretrained=True)
param = resnet_model.state_dict()
for i in param.keys():
  print(i)

"""# Function to Finetune a Single Layer"""

def selective_finetuning_single_layer(layer_name):
  resnet_model = models.resnet50(pretrained=True)
  for name, param in resnet_model.named_parameters():
    if param.requires_grad and layer_name in name:
      param.requires_grad = True
    else:
      param.requires_grad = False
  resnet_model.fc = torch.nn.Linear(2048,10)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()))
  return resnet_model.cuda(), optimizer

loss_function_fine_tune = torch.nn.CrossEntropyLoss().cuda()
resnet_finetune_1, optimizer_1 = selective_finetuning_single_layer('layer1')
resnet_finetune_2, optimizer_2 = selective_finetuning_single_layer('layer2')
resnet_finetune_3, optimizer_3 = selective_finetuning_single_layer('layer3')
resnet_finetune_4, optimizer_4 = selective_finetuning_single_layer('layer4')

resnet_finetune_1,output_train_finetune_1,output_test_finetune_1 = model(train,test,15,resnet_finetune_1,loss_function_fine_tune,optimizer_1)
resnet_finetune_2,output_train_finetune_2,output_test_finetune_2 = model(train,test,15,resnet_finetune_2,loss_function_fine_tune,optimizer_2)
resnet_finetune_3,output_train_finetune_3,output_test_finetune_3 = model(train,test,15,resnet_finetune_3,loss_function_fine_tune,optimizer_3)
resnet_finetune_4,output_train_finetune_4,output_test_finetune_4 = model(train,test,15,resnet_finetune_4,loss_function_fine_tune,optimizer_4)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_default[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_1[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_2[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_3[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_4[i][1] for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss v/s Epochs for Training with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Layer-1 Fine-tuning","Layer-2 Fine-tuning","Layer-3 Fine-tuning","Layer-4 Fine-tuning"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_test_default[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_1[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_2[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_3[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_4[i][1] for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss v/s Epochs for Testing with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Layer-1 Fine-tuning","Layer-2 Fine-tuning","Layer-3 Fine-tuning","Layer-4 Fine-tuning"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_default[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_1[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_2[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_3[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_4[i][2].cpu().numpy() for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs for Training with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Layer-1 Fine-tuning","Layer-2 Fine-tuning","Layer-3 Fine-tuning","Layer-4 Fine-tuning"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_test_default[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_1[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_2[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_3[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_4[i][2].cpu().numpy() for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs for Testing with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Layer-1 Fine-tuning","Layer-2 Fine-tuning","Layer-3 Fine-tuning","Layer-4 Fine-tuning"])
plt.show()

"""# Function to Finetune multiple layers"""

def selective_finetuning_multiple_layers(layer_list):
  resnet_model = models.resnet50(pretrained=True)
  for name, param in resnet_model.named_parameters():
    for layer_name in layer_list:
      if param.requires_grad and layer_name in name:
        param.requires_grad = True
      else:
        param.requires_grad = False
  resnet_model.fc = torch.nn.Linear(2048,10)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()))
  return resnet_model.cuda(), optimizer

resnet_finetune_conv, optimizer_conv = selective_finetuning_multiple_layers(['conv1','conv2','conv3'])
resnet_finetune_bn, optimizer_bn = selective_finetuning_multiple_layers(['bn1','bn2','bn3'])
resnet_finetune_downsample, optimizer_downsample = selective_finetuning_multiple_layers('downsample')

resnet_finetune_conv,output_train_finetune_conv,output_test_finetune_conv = model(train,test,15,resnet_finetune_conv,loss_function_fine_tune,optimizer_conv)
resnet_finetune_bn,output_train_finetune_bn,output_test_finetune_bn = model(train,test,15,resnet_finetune_bn,loss_function_fine_tune,optimizer_bn)
resnet_finetune_downsample,output_train_finetune_downsample,output_test_finetune_downsample = model(train,test,15,resnet_finetune_downsample,loss_function_fine_tune,optimizer_downsample)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_default[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_conv[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_bn[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_downsample[i][1] for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss v/s Epochs for Training with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Conv Layers Fine-tuning","Batch Normalization Layers Fine-tuning","Downsampling Layers Fine-tuning"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_test_default[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_conv[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_bn[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_downsample[i][1] for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss v/s Epochs for Testing with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Conv Layers Fine-tuning","Batch Normalization Layers Fine-tuning","Downsampling Layers Fine-tuning"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_default[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_conv[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_bn[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_train_finetune_downsample[i][2].cpu().numpy() for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs for Training with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Conv Layers Fine-tuning","Batch Normalization Layers Fine-tuning","Downsampling Layers Fine-tuning"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_test_default[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_conv[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_bn[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_test_finetune_downsample[i][2].cpu().numpy() for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs for Testing with variation in layer fine-tuning")
plt.legend(["Base Network Frozen","Conv Layers Fine-tuning","Batch Normalization Layers Fine-tuning","Downsampling Layers Fine-tuning"])
plt.show()

"""# Function to calculate Confusion Matrix, Class-Wise Accuracy and Mean-Classwise Accuracy"""

def second_question_subparts(model_class, test_loader):
  pred = []
  true = []
  with torch.no_grad():
    for (image, label) in test_loader:
      image = image.cuda()
      label = label.cuda()
      for jj in label:
        true.append(jj.cpu().item())
      out = model_class(image)
      predicted_test = out.data.max(1, keepdim=True)[1]
      for j in predicted_test:
        pred.append(j.cpu().item())
  length = max(true)
  confusion_matrix = [[0 for i in range(length)] for j in range(length)]
  for row in range(0,len(true)):
    column = pred[row]
    confusion_matrix[true[row]-1][pred[row]-1] +=1
  print("Confusion Matrix\n")
  for i in confusion_matrix:
    print(i)
  mean_acc = 0
  print("\nClass-Wise Accuracy\n")
  for idx in range(length):
    tp = confusion_matrix[idx][idx]
    overall = sum(confusion_matrix[idx])
    output = tp/overall
    mean_acc += output
    print('For Class-'+str(idx)+' :- '+str(output*100)+'%\n')
  print('Mean Class-Wise Accuracy :-'+str(mean_acc*100/length)+'%\n')
  correct=0
  for i in range(length):
    correct += confusion_matrix[i][i]
  print('Overall Accuracy :-'+str(correct*100/len(true))+"%")

second_question_subparts(resnet_model_default, test)

second_question_subparts(resnet_finetune_1, test)

second_question_subparts(resnet_finetune_2, test)

second_question_subparts(resnet_finetune_3, test)

second_question_subparts(resnet_finetune_4, test)

second_question_subparts(resnet_finetune_conv, test)

second_question_subparts(resnet_finetune_bn, test)

second_question_subparts(resnet_finetune_downsample, test)
