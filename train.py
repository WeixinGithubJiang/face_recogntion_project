from gen_dataset import Dataset
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50
# from net import Net
from utils import RandomCrop, RandomResize,Preprocess,ProgressBar
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import time
import h5py
import os
import torch.nn.functional as F
# %matplotlib inline


    
def model_train(model,train_loader,optimizer,device):
    model.train()
    progress_bar = ProgressBar(len(train_loader))
    for i, data in enumerate(train_loader,0):
        model.zero_grad()
        optimizer.zero_grad()
        
        # inputs
        inpt = data['faces']
        labels = data['labels']
        target = torch.LongTensor(labels).to(device)
        inpt_train = torch.Tensor(inpt).to(device)
        target.requires_grad = False
        inpt_train.requires_grad = False
        
        # predict
        out_train = model(inpt_train)
        loss = F.nll_loss(out_train,target)
#         loss = criterion(out_train,target)
        loss.backward()
        optimizer.step()
        
        progress_bar.update(i,loss.data.cpu().numpy())
        
        
def save_model(model,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model.state_dict(),os.path.join(save_dir,'net.pth'))
        
if __name__ == '__main__':
    
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
        
    hf = h5py.File('./dataset.h5', 'r')
    faces = np.array(hf.get('faces_train'))/255
    labels = np.array(hf.get('labels_train'))
    hf.close()
    
    batch_size = 32
    train_dataset = Dataset(faces,labels)
    train_loader = DataLoader(dataset=train_dataset, num_workers=6, batch_size=batch_size, shuffle=True)

    net = resnet18(num_classes = 142)
    model = net.to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    
    max_epoch = 100
    for epoch in range(max_epoch):  
        new_faces = Preprocess(faces)
        train_dataset = Dataset(new_faces,labels)
        train_loader = DataLoader(dataset=train_dataset, num_workers=6, batch_size=batch_size, shuffle=True)
        model_train(model,train_loader,optimizer,device)
        print("%d/%d is completed!!!" % (epoch+1,max_epoch))
    
    save_dir = './saved_model'
    save_model(model,save_dir)