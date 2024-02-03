#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os


# In[2]:


class Dataset(Dataset):

    def __init__(self,transform=None,train=True):
        q=os.getcwd()+'\concrete_crack_images_for_classification'
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(q,positive)
        negative_file_path=os.path.join(q,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()
        number_of_samples=len(positive_files)+len(negative_files)
        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        self.transform = transform
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:10000] #Change to 30000 to use the full test dataset
            self.Y=self.Y[0:10000] #Change to 30000 to use the full test dataset
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)    
       
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]
          
        
        if self.transform:
            image = self.transform(image)

        return image, y


# In[3]:


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])


# In[4]:


dataset_train=Dataset(transform=transform,train=True)
dataset_val=Dataset(transform=transform,train=False)


# In[6]:


model=models.resnet18(pretrained=True)


# In[7]:


for params in model.parameters():
    params.requires_grad=False


# In[8]:


model.fc=nn.Linear(512,2)


# In[9]:


print(model)


# In[12]:


criterion=nn.CrossEntropyLoss()


# In[11]:


train_loader=DataLoader(dataset=dataset_train,batch_size=100)
val_loader=DataLoader(dataset=dataset_val,batch_size=100)


# In[13]:


optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad],lr=0.001)


# In[20]:


n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(dataset_val)
N_train=len(dataset_train)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:

        model.train() 
        optimizer.zero_grad()
        z=model(x)
        loss=criterion(z,y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
    correct=0
    for x_test, y_test in val_loader:
        model.eval()
        z=model(x_test)
        _,yhat=torch.max(z.data,1)
        correct+=(yhat==y_test).sum().item()
    accuracy_list.append(correct/N_test)
        


# In[21]:


accuracy_list


# In[22]:


loss_list


# In[19]:


correct


# In[26]:


plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()


# In[29]:


#Identify the first four misclassified samples using the validation data:

n=0
s=0
for i,j in val_loader:
    model.eval()
    z=model(i)
    _,yhat=torch.max(z.data,1)
    for x in range(len(j)):
        s+=1
        if yhat[x]!=j[x]:
            
            print("sample#: %d - predicted value: %d - actual value: %d" % (s, yhat[x], y_test[x]))
            n+=1
            if n>=4:
                break
    if n>=4:
        break
print('Done')
    


# In[ ]:




