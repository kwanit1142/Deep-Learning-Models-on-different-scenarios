# -*- coding: utf-8 -*-
"""B19EE046_DL_Assignment_2_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SpzCOA4UbdYFI4MxjYNLT0ENYUxj1rz5
"""

from google.colab import drive
drive.mount('/content/drive')

"""# Importing Necessary Libraries"""

import numpy as np
import torch
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from random import shuffle
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder as LE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""# Dataset Handling

## Fetching Dataset from Web-Link
"""

!wget 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' -P '/content/drive/MyDrive/DL_Assignment_2/'

"""## Unzipping the Dataset File"""

!unzip '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200.zip' -d '/content/drive/MyDrive/DL_Assignment_2/'

"""## Utility Function to save meta-file (Images + Labels)"""

def save_train_meta(directory):
  df = pd.DataFrame(columns=['Image_Name','Image_Class'])
  img_name_list=[]
  img_class_list=[]
  class_list = os.listdir(directory)
  for class_name in tqdm(class_list):
    folder_directory = os.path.join(directory,class_name)
    folder_list = os.listdir(folder_directory+'/images')
    for image_name in folder_list:
      img_name_list.append(image_name)
      img_class_list.append(class_name)
  df['Image_Name']=img_name_list
  df['Image_Class']=img_class_list
  save_directory = directory+'_meta.csv'
  df.to_csv(save_directory)

save_train_meta('/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/train')

"""## Meta-File with Label-Encoding"""

train_df = pd.read_csv('/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/train_meta.csv')
train_df = train_df.drop(columns=['Unnamed: 0'])
le = LE().fit(train_df['Image_Class']) 
train_df['Class_Encoded']=le.transform(train_df['Image_Class'])

val_df = pd.read_csv('/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/val/val_annotations.txt', sep = '\t', header = None)
val_df.columns = ['Image_Name','Image_Class','x1','x2','y1','y2']
val_df = val_df.drop(columns=['x1','x2','y1','y2'])
le_2 = LE().fit(val_df['Image_Class'])
val_df['Class_Encoded']=le_2.transform(val_df['Image_Class'])

train_df = train_df[train_df.Class_Encoded<=49]
train_df = train_df.reset_index(drop=True)
val_df = val_df[val_df.Class_Encoded<=49]
val_df = val_df.reset_index(drop=True)

"""## Dataset-Class and Utility Function to construct a Pytorch-compatible  Dataloader"""

class Train_Prepare(Dataset):
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
    img_name = os.path.join(self.root_dir, str(self.data_frame.iloc[idx, 1])+'/images/'+str(self.data_frame.iloc[idx, 0]))
    image = cv2.imread(img_name)
    label = self.data_frame.iloc[idx, -1]
    if self.transform:
      image = self.transform(image)
    return (image, label)

class Val_Prepare(Dataset):
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
    img_name = os.path.join(self.root_dir, 'images/'+str(self.data_frame.iloc[idx, 0]))
    image = cv2.imread(img_name)
    label = self.data_frame.iloc[idx, -1]
    if self.transform:
      image = self.transform(image)
    return (image, label)

def data_preparation(Data_Class_1, Data_Class_2, root_directory_train, root_directory_val, train_df, val_df, Batch_Size = 128, Shuffle = False):
  train_dataset = Data_Class_1(data_frame=train_df,root_dir=root_directory_train, transform = transforms.ToTensor())
  train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Batch_Size, shuffle = Shuffle, pin_memory=True, num_workers=2)
  val_dataset = Data_Class_2(data_frame=val_df,root_dir=root_directory_val, transform = transforms.ToTensor())
  val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = Batch_Size, shuffle = Shuffle, pin_memory=True, num_workers=2)
  return train_loader, val_loader

train, val = data_preparation(Train_Prepare, Val_Prepare, '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/train', '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/val', train_df, val_df, Batch_Size = 400, Shuffle = True)

"""# Utility Function to train a Pytorch Model"""

def grad_change(Loss_Function, Optimizer, Label = None, Predicted = None):
  Optimizer.zero_grad()
  loss = Loss_Function(Predicted, Label)
  loss.backward()
  Optimizer.step()
  return loss, Optimizer

def model(Train_Loader, Test_Loader, Epochs, Model_Class=None, Loss_Function=None, Optimizer=None):
  outputs_train=[]
  outputs_test=[]
  for Epoch in tqdm(range(Epochs)):
    running_loss_train=0
    running_loss_test=0
    correct_train=0
    correct_test=0
    count=0
    for (image, label) in Train_Loader:
      count+=1
      print(count)
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

"""# 1.) Optimization via Cross-Entropy Loss"""

resnet_model_cross = models.resnet18(pretrained=True).cuda()
loss_cross = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(resnet_model_cross.parameters())

resnet_1,output_train_1,output_val_1 = model(train,val,15,resnet_model_cross,loss_cross,optimizer)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_1[i][1] for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_val_1[i][1] for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss v/s Epochs Curve")
plt.legend(["Train","Val"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,16)],[output_train_1[i][2].cpu().numpy() for i in range(0,15)])
plt.plot([j for j in range(1,16)],[output_val_1[i][2].cpu().numpy() for i in range(0,15)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve")
plt.legend(["Train","Val"])
plt.show()

"""# 2.) Optimization via Center Loss"""

train, val = data_preparation(Train_Prepare, Val_Prepare, '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/train', '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/val', train_df, val_df, Batch_Size = 500, Shuffle = False)

"""## Pytorch-Compatible Class for Center-Loss"""

class CenterLoss(torch.nn.Module):
  
  def __init__(self, num_classes=10, feat_dim=2, batch_user = 128):
    super(CenterLoss, self).__init__()
    self.num_classes = num_classes
    self.feat_dim = feat_dim
    self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
    self.batch_size=batch_user

  def forward(self, x, labels):
    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(self.batch_size, self.num_classes) + \
              torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, self.batch_size).t()
    distmat.addmm_(1, -2, x, self.centers.t())
    classes = torch.arange(self.num_classes).long().cuda()
    labels = labels.unsqueeze(1).expand(self.batch_size, self.num_classes)
    mask = labels.eq(classes.expand(self.batch_size, self.num_classes))
    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / (2*self.batch_size)
    return loss

"""## Pytorch-Compatible Class for Feature-Extractor"""

class new_model(torch.nn.Module):
  def __init__(self, model_class, output_layer):
    super().__init__()
    self.output_layer = output_layer
    self.pretrained = model_class
    self.children_list = []
    for n,c in self.pretrained.named_children():
      self.children_list.append(c)
      if n == self.output_layer:
        break
    self.children_final = self.children_list[:-1]
    self.net = torch.nn.Sequential(*self.children_final)
    self.pretrained = None
      
  def forward(self,x):
    count=0
    for layer in self.net:
      count+=1
      if count==10:
        x = x.view(-1,512*1*1)
      x = layer(x)
    return x

"""## ResNet Model with n_dim=2 and batch_size=500 for Default Configurations"""

resnet_model_center = models.resnet18(pretrained=True)
n_input = resnet_model_center.fc.out_features
resnet_model_center.fc1 = torch.nn.Linear(n_input,2)
resnet_model_center.fc1_activation = torch.nn.LeakyReLU()
resnet_model_center.fc2 = torch.nn.Linear(2,50)
resnet_model_center.cuda()
model_loss = torch.nn.CrossEntropyLoss().cuda()
center_loss = CenterLoss(50, 2, 500)
params_center = list(resnet_model_center.parameters()) + list(center_loss.parameters())
optimizer_center = torch.optim.Adam(params_center, lr=0.01)

"""## Utility Function to train Pytorch-Model with varied Learning Rate, Alpha and Central-Learning Rate"""

def grad_change_center_lr(Loss_Function_model, Loss_Function_center, Optimizer, Label = None, Predicted = None, Features = None, alpha_chosen=0.1, lr_cent_user = 0.5, lr_optim_user = 0.1):
  cross_loss = Loss_Function_model(Predicted, Label)/500
  center_loss = Loss_Function_center(Features, Label)
  overall_loss = cross_loss + center_loss*alpha_chosen
  Optimizer.zero_grad()
  overall_loss.backward()
  for param in Loss_Function_center.parameters():
    param.grad.data *= (lr_cent_user / (alpha_chosen * lr_optim_user))
  Optimizer.step()
  return overall_loss, center_loss, cross_loss, Optimizer

def model_center_lr(Train_Loader, Test_Loader, Epochs, Model_Class=None, Model_Features=None, Loss_Function_model=None, Loss_Function_center=None, Optimizer=None, alpha=0.1, lr_cent=0.5, lr_optim = 0.1):
  outputs_train=[]
  outputs_test=[]
  for Epoch in tqdm(range(Epochs)):
    running_loss_train=0
    center_loss_train=0
    cross_loss_train=0
    running_loss_test=0
    center_loss_test=0
    cross_loss_test=0
    correct_train=0
    correct_test=0
    for (image, label) in Train_Loader:
      image = image.cuda()
      label = label.cuda()
      output_pred = Model_Class(image)
      model_feat = Model_Features(model_class = Model_Class, output_layer = 'fc1_activation')
      output_feat = model_feat(image)
      loss, cent, cross, Optimizer = grad_change_center_lr(Loss_Function_model = Loss_Function_model, Loss_Function_center = Loss_Function_center, Optimizer = Optimizer, Label = label, Predicted = output_pred, Features = output_feat, alpha_chosen=alpha, lr_cent_user = lr_cent, lr_optim_user = lr_optim)
      running_loss_train += loss.item()
      center_loss_train += cent.item()
      cross_loss_train += cross.item()
      predicted_train = output_pred.data.max(1, keepdim=True)[1]
      correct_train += predicted_train.eq(label.data.view_as(predicted_train)).sum()
      print(running_loss_train, center_loss_train, cross_loss_train)
    outputs_train.append((Epoch, running_loss_train, center_loss_train, cross_loss_train,100*correct_train/len(Train_Loader.dataset)))
    with torch.no_grad():
      for (image, label) in Test_Loader:
        image = image.cuda()
        label = label.cuda()
        out_pred = Model_Class(image)
        mod_feat = Model_Features(model_class = Model_Class, output_layer = 'fc1_activation')
        out_feat = mod_feat(image)
        cross_loss = Loss_Function_model(out_pred,label)/500
        cent_loss = Loss_Function_center(out_feat,label)
        overall_loss = cross_loss + cent_loss*alpha
        running_loss_test += overall_loss.item()
        center_loss_test += cent.item()
        cross_loss_test += cross.item()
        predicted_test = out_pred.data.max(1, keepdim=True)[1]
        correct_test += predicted_test.eq(label.data.view_as(predicted_test)).sum()
        print(running_loss_test, center_loss_test, cross_loss_test)
      outputs_test.append((Epoch, running_loss_test, center_loss_test, cross_loss_test, 100*correct_test/len(Test_Loader.dataset)))
  return Model_Class, outputs_train, outputs_test

"""## For Alpha=0.1"""

resnet_model_center_1,output_train_center_1,output_val_center_1 = model_center_lr(train,val,20,resnet_model_center,new_model,model_loss,center_loss,optimizer_center,alpha=0.1, lr_cent=0.5, lr_optim=0.01)

"""## For Alpha = 0.01"""

resnet_model_center_2 = models.resnet18(pretrained=True)
resnet_model_center_2.fc1 = torch.nn.Linear(n_input,2)
resnet_model_center_2.fc1_activation = torch.nn.LeakyReLU()
resnet_model_center_2.fc2 = torch.nn.Linear(2,50)
resnet_model_center_2.cuda()
center_loss_2 = CenterLoss(50, 2, 500)
params_center_2 = list(resnet_model_center_2.parameters()) + list(center_loss_2.parameters())
optimizer_center_2 = torch.optim.Adam(params_center_2, lr=0.01)
resnet_model_center_2,output_train_center_2,output_val_center_2 = model_center_lr(train,val,20,resnet_model_center_2,new_model,model_loss,center_loss_2,optimizer_center_2,alpha=0.01, lr_cent=0.5, lr_optim=0.01)

"""## For Alpha = 0.001"""

resnet_model_center_3 = models.resnet18(pretrained=True)
resnet_model_center_3.fc1 = torch.nn.Linear(n_input,2)
resnet_model_center_3.fc1_activation = torch.nn.LeakyReLU()
resnet_model_center_3.fc2 = torch.nn.Linear(2,50)
resnet_model_center_3.cuda()
center_loss_3 = CenterLoss(50, 2, 500)
params_center_3 = list(resnet_model_center_3.parameters()) + list(center_loss_3.parameters())
optimizer_center_3 = torch.optim.Adam(params_center_3, lr=0.01)
resnet_model_center_3,output_train_center_3,output_val_center_3 = model_center_lr(train,val,20,resnet_model_center_3,new_model,model_loss,center_loss_3,optimizer_center_3,alpha=0.001, lr_cent=0.5, lr_optim=0.01)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_1[i][1] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_2[i][1] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_3[i][1] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Overall Loss")
plt.title("Overall Loss v/s Epochs Curve for Training in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_1[i][1] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_2[i][1] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_3[i][1] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Overall Loss")
plt.title("Overall Loss v/s Epochs Curve for Testing in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_1[i][2] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_2[i][2] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_3[i][2] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Center Loss")
plt.title("Center Loss v/s Epochs Curve for Training in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_1[i][2] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_2[i][2] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_3[i][2] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Center Loss")
plt.title("Center Loss v/s Epochs Curve for Testing in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_1[i][3] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_2[i][3] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_3[i][3] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Cross-Entropy Loss v/s Epochs Curve for Training in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_1[i][3] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_2[i][3] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_3[i][3] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Cross-Entropy Loss v/s Epochs Curve for Testing in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_1[i][4].cpu().numpy() for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_2[i][4].cpu().numpy() for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_3[i][4].cpu().numpy() for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve for Training in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_1[i][4].cpu().numpy() for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_2[i][4].cpu().numpy() for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_3[i][4].cpu().numpy() for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve for Testing in variation to alpha")
plt.legend(["0.1","0.01","0.001"])
plt.show()

"""## For lr_cent = 0.4 and Alpha = 0.1"""

resnet_model_center_0_4 = models.resnet18(pretrained=True)
n_input = resnet_model_center_0_4.fc.out_features
resnet_model_center_0_4.fc1 = torch.nn.Linear(n_input,2)
resnet_model_center_0_4.fc1_activation = torch.nn.LeakyReLU()
resnet_model_center_0_4.fc2 = torch.nn.Linear(2,50)
resnet_model_center_0_4.cuda()
model_loss = torch.nn.CrossEntropyLoss().cuda()
center_loss_0_4 = CenterLoss(50, 2, 500)
params_center_0_4 = list(resnet_model_center_0_4.parameters()) + list(center_loss_0_4.parameters())
optimizer_center_0_4 = torch.optim.Adam(params_center_0_4, lr=0.005)
resnet_model_center_0_4,output_train_center_0_4,output_val_center_0_4 = model_center_lr(train,val,20,resnet_model_center_0_4,new_model,model_loss,center_loss_0_4,optimizer_center_0_4,alpha=0.1, lr_cent=0.4, lr_optim=0.01)

"""## For lr_cent = 0.3 and Alpha = 0.1"""

resnet_model_center_0_3 = models.resnet18(pretrained=True)
resnet_model_center_0_3.fc1 = torch.nn.Linear(n_input,2)
resnet_model_center_0_3.fc1_activation = torch.nn.LeakyReLU()
resnet_model_center_0_3.fc2 = torch.nn.Linear(2,50)
resnet_model_center_0_3.cuda()
center_loss_0_3 = CenterLoss(50, 2, 500)
params_center_0_3 = list(resnet_model_center_0_3.parameters()) + list(center_loss_0_3.parameters())
optimizer_center_0_3 = torch.optim.Adam(params_center_0_3, lr=0.005)
resnet_model_center_0_3,output_train_center_0_3,output_val_center_0_3 = model_center_lr(train,val,20,resnet_model_center_0_3,new_model,model_loss,center_loss_0_3,optimizer_center_0_3,alpha=0.1, lr_cent=0.3, lr_optim=0.01)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_0_4[i][1] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_0_3[i][1] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Overall Loss")
plt.title("Overall Loss v/s Epochs Curve for Training in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_0_4[i][1] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_0_3[i][1] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Overall Loss")
plt.title("Overall Loss v/s Epochs Curve for Testing in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_0_4[i][2] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_0_3[i][2] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Center Loss")
plt.title("Center Loss v/s Epochs Curve for Training in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_0_4[i][2] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_0_3[i][2] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Center Loss")
plt.title("Center Loss v/s Epochs Curve for Testing in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_0_4[i][3] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_0_3[i][3] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Cross-Entropy Loss v/s Epochs Curve for Training in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_0_4[i][3] for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_0_3[i][3] for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Cross-Entropy Loss v/s Epochs Curve for Testing in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_train_center_0_4[i][4].cpu().numpy() for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_train_center_0_3[i][4].cpu().numpy() for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve for Training in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,21)],[output_val_center_0_4[i][4].cpu().numpy() for i in range(0,20)])
plt.plot([j for j in range(1,21)],[output_val_center_0_3[i][4].cpu().numpy() for i in range(0,20)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve for Testing in variation to lr_cent")
plt.legend(["0.4","0.3"])
plt.show()

"""# 3.) Optimization via Triplet Loss"""

train_triplet, val_triplet = data_preparation(Train_Prepare, Val_Prepare, '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/train', '/content/drive/MyDrive/DL_Assignment_2/tiny-imagenet-200/val', train_df, val_df, Batch_Size = 500, Shuffle = True)

"""## Pytorch-Compatible Class for Triplet Loss"""

class Triplet_Loss(torch.nn.Module):
  def __init__(self, margin_alpha):
    super(Triplet_Loss, self).__init__()
    self.margin = margin_alpha
    self.dist_p = 1000
    self.dist_n = 1000
    self.loss_sum = 0
  
  def dist_calc(self,x1,x2):
    x1_ed = (x1[0].item() - x2[0].item())**2
    x2_ed = (x1[1].item() - x2[1].item())**2
    return x1_ed + x2_ed

  def forward(self, Feature_Matrix, Label_Vector):
    for i in range(0,Label_Vector.shape[0]-1):
      for j in range(i+1,Label_Vector.shape[0]):
        dist = self.dist_calc(Feature_Matrix[i],Feature_Matrix[j])
        if (Label_Vector[i]==Label_Vector[j]) and (self.dist_p > dist):
          self.dist_p = dist
        if (Label_Vector[i]!=Label_Vector[j]) and (self.dist_n > dist):
          self.dist_n = dist
      if (self.dist_p - self.dist_n + self.margin)>0:
        self.loss_sum += self.dist_p - self.dist_n + self.margin
      else:
        self.loss_sum += 0
    loss = self.loss_sum
    return loss

"""## For Margin = 0.5"""

resnet_model_triplet = models.resnet18(pretrained=True)
n_input = resnet_model_triplet.fc.out_features
resnet_model_triplet.fc1 = torch.nn.Linear(n_input,2)
resnet_model_triplet.fc1_activation = torch.nn.LeakyReLU()
resnet_model_triplet.fc2 = torch.nn.Linear(2,50)
resnet_model_triplet.cuda()
model_loss = torch.nn.CrossEntropyLoss().cuda()
triplet_loss = Triplet_Loss(0.5)
params_triplet = list(resnet_model_triplet.parameters()) + list(triplet_loss.parameters())
optimizer_triplet = torch.optim.Adam(params_triplet, lr=0.01)

"""## Utility Functions to train a Pytorch-Model with varied Margin and Learning Rate"""

def grad_change_triplet_lr(Loss_Function_model, Loss_Function_triplet, Optimizer, Label = None, Predicted = None, Features = None, lambda_chosen=0.1, lr_triplet_user = 0.5, lr_optim_user = 0.1):
  cross_loss = Loss_Function_model(Predicted, Label)/500
  triplet_loss = Loss_Function_triplet(Features, Label)/500
  overall_loss = cross_loss + triplet_loss*lambda_chosen
  Optimizer.zero_grad()
  overall_loss.backward()
  for param in Loss_Function_triplet.parameters():
    param.grad.data *= (lr_triplet_user / (lambda_chosen * lr_optim_user))
  Optimizer.step()
  return overall_loss, triplet_loss, cross_loss, Optimizer

def model_triplet_lr(Train_Loader, Test_Loader, Epochs, Model_Class=None, Model_Features=None, Loss_Function_model=None, Loss_Function_triplet=None, Optimizer=None, lambda_param=0.1, lr_triplet=0.5, lr_optim = 0.1):
  outputs_train=[]
  outputs_test=[]
  for Epoch in tqdm(range(Epochs)):
    running_loss_train=0
    triplet_loss_train=0
    cross_loss_train=0
    running_loss_test=0
    triplet_loss_test=0
    cross_loss_test=0
    correct_train=0
    correct_test=0
    for (image, label) in Train_Loader:
      image = image.cuda()
      label = label.cuda()
      output_pred = Model_Class(image)
      model_feat = Model_Features(model_class = Model_Class, output_layer = 'fc1_activation')
      output_feat = model_feat(image)
      loss, cent, cross, Optimizer = grad_change_triplet_lr(Loss_Function_model = Loss_Function_model, Loss_Function_triplet = Loss_Function_triplet, Optimizer = Optimizer, Label = label, Predicted = output_pred, Features = output_feat, lambda_chosen=lambda_param, lr_triplet_user = lr_triplet, lr_optim_user = lr_optim)
      running_loss_train += loss.item()
      triplet_loss_train += cent
      cross_loss_train += cross.item()
      predicted_train = output_pred.data.max(1, keepdim=True)[1]
      correct_train += predicted_train.eq(label.data.view_as(predicted_train)).sum()
      print(running_loss_train, triplet_loss_train, cross_loss_train)
    outputs_train.append((Epoch, running_loss_train, triplet_loss_train, cross_loss_train,100*correct_train/len(Train_Loader.dataset)))
    with torch.no_grad():
      for (image, label) in Test_Loader:
        image = image.cuda()
        label = label.cuda()
        out_pred = Model_Class(image)
        mod_feat = Model_Features(model_class = Model_Class, output_layer = 'fc1_activation')
        out_feat = mod_feat(image)
        cross_loss = Loss_Function_model(out_pred,label)
        cent_loss = Loss_Function_triplet(out_feat,label)
        overall_loss = cross_loss + cent_loss*lambda_param
        running_loss_test += overall_loss.item()
        triplet_loss_test += cent
        cross_loss_test += cross.item()
        predicted_test = out_pred.data.max(1, keepdim=True)[1]
        correct_test += predicted_test.eq(label.data.view_as(predicted_test)).sum()
        print(running_loss_test, triplet_loss_test, cross_loss_test)
      outputs_test.append((Epoch, running_loss_test/len(Test_Loader.dataset), triplet_loss_test/len(Test_Loader.dataset), cross_loss_test/len(Test_Loader.dataset), 100*correct_test/len(Test_Loader.dataset)))
  return Model_Class, outputs_train, outputs_test

resnet_model_triplet_1,output_train_triplet_1,output_val_triplet_1 = model_triplet_lr(train_triplet,val_triplet,5,resnet_model_triplet,new_model,model_loss,triplet_loss,optimizer_triplet,lambda_param=0.1, lr_triplet=0.5, lr_optim=0.01)

"""## For Margin = 1"""

resnet_model_triplet_2 = models.resnet18(pretrained=True)
n_input = resnet_model_triplet_2.fc.out_features
resnet_model_triplet_2.fc1 = torch.nn.Linear(n_input,2)
resnet_model_triplet_2.fc1_activation = torch.nn.LeakyReLU()
resnet_model_triplet_2.fc2 = torch.nn.Linear(2,50)
resnet_model_triplet_2.cuda()
model_loss = torch.nn.CrossEntropyLoss().cuda()
triplet_loss_2 = Triplet_Loss(1)
params_triplet_2 = list(resnet_model_triplet_2.parameters()) + list(triplet_loss_2.parameters())
optimizer_triplet_2 = torch.optim.Adam(params_triplet_2, lr=0.01)
resnet_model_triplet_2,output_train_triplet_2,output_val_triplet_2 = model_triplet_lr(train_triplet,val_triplet,5,resnet_model_triplet_2,new_model,model_loss,triplet_loss_2,optimizer_triplet_2,lambda_param=0.1, lr_triplet=0.5, lr_optim=0.01)

"""## For Margin = 1.5"""

resnet_model_triplet_3 = models.resnet18(pretrained=True)
n_input = resnet_model_triplet_3.fc.out_features
resnet_model_triplet_3.fc1 = torch.nn.Linear(n_input,2)
resnet_model_triplet_3.fc1_activation = torch.nn.LeakyReLU()
resnet_model_triplet_3.fc2 = torch.nn.Linear(2,50)
resnet_model_triplet_3.cuda()
model_loss = torch.nn.CrossEntropyLoss().cuda()
triplet_loss_3 = Triplet_Loss(1.5)
params_triplet_3 = list(resnet_model_triplet_3.parameters()) + list(triplet_loss_3.parameters())
optimizer_triplet_3 = torch.optim.Adam(params_triplet_3, lr=0.01)
resnet_model_triplet_3,output_train_triplet_3,output_val_triplet_3 = model_triplet_lr(train_triplet,val_triplet,5,resnet_model_triplet_3,new_model,model_loss,triplet_loss_3,optimizer_triplet_3,lambda_param=0.1, lr_triplet=0.5, lr_optim=0.01)

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_train_triplet_1[i][1] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_2[i][1] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_3[i][1] for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Average Overall Loss")
plt.title("Overall Loss v/s Epochs Curve for Training in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_val_triplet_1[i][1] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_2[i][1] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_3[i][1] for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Average Overall Loss")
plt.title("Overall Loss v/s Epochs Curve for Testing in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_train_triplet_1[i][2] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_2[i][2] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_3[i][2] for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Average Triplet Loss")
plt.title("Triplet Loss v/s Epochs Curve for Training in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_val_triplet_1[i][2] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_2[i][2] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_3[i][2] for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Average Triplet Loss")
plt.title("Triplet Loss v/s Epochs Curve for Testing in variation to margins")
plt.legend(["0.1","0.05","0.15"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_train_triplet_1[i][3] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_2[i][3] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_3[i][3] for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Cross-Entropy Loss v/s Epochs Curve for Training in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_val_triplet_1[i][3] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_2[i][3] for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_3[i][3] for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("Cross-Entropy Loss v/s Epochs Curve for Testing in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_train_triplet_1[i][4].cpu().numpy() for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_2[i][4].cpu().numpy() for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_train_triplet_3[i][4].cpu().numpy() for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve for Training in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()

plt.figure(figsize=(10,5))
plt.plot([j for j in range(1,6)],[output_val_triplet_1[i][4].cpu().numpy() for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_2[i][4].cpu().numpy() for i in range(0,5)])
plt.plot([j for j in range(1,6)],[output_val_triplet_3[i][4].cpu().numpy() for i in range(0,5)])
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy v/s Epochs Curve for Testing in variation to margins")
plt.legend(["0.5","1","1.5"])
plt.show()