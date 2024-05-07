import numpy as np
import torch
import os
import time

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam

from utils.imgprocess import img_transform_512,img_transform_256,load_image
from utils.loss import gram

from net.vgg16 import Vgg16
from net.transform import TransformNet

style_path = "styleimg/NewMediaArt_style.jpg"
#NewMediaArt_style.jpg
dataset_path = "E:\\AI\\train2014"
bs = 4
epochs = 2
dtype = torch.cuda.FloatTensor
style_weight = 1e5
content_weight = 1e0
tv_weight =  1e-7


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = TransformNet().type(dtype)
vgg = Vgg16().type(dtype)

train_dataset = datasets.ImageFolder(dataset_path, img_transform_256)
train_loader = DataLoader(train_dataset, batch_size = bs)

style_dataset = load_image(style_path)
style_dataset = img_transform_256(style_dataset)
style_dataset = Variable(style_dataset.repeat(bs, 1, 1, 1)).type(dtype)
style_features = vgg(style_dataset)
style_gram = [gram(fmap) for fmap in style_features]

optimizer = Adam(transformer.parameters(), 1e-3) 
loss_mse = torch.nn.MSELoss()

for ep in range(epochs):
        img_count = 0
        total_style_loss = 0.0
        total_content_loss = 0.0
        total_tv_loss = 0.0

        transformer.train()
        for batch_num , (x,label) in  enumerate(train_loader):
            img_batch_count = len(x)
            img_count+=img_batch_count

            optimizer.zero_grad()
            x = Variable(x).type(dtype)
            x_transform = transformer(x)

            x_features = vgg(x)
            x_transform_features = vgg(x_transform)

            #Style Loss
            x_transform_gram = [gram(fmap) for fmap in x_transform_features]
            style_loss=content_loss=tv_loss = 0.0

            for j in range(4):
                style_loss += loss_mse(x_transform_gram[j], style_gram[j])

            style_loss = style_weight*style_loss
            total_style_loss += style_loss.item()

            #Content Loss
            x_content_feature = x_features[1]
            x_transform_content_feature = x_transform_features[1]
            content_loss = content_weight*loss_mse(x_transform_content_feature, x_content_feature)
            total_content_loss += content_loss.item()

            #Total variation Loss
            diff_x = torch.sum(torch.abs(x_transform[:, :, :, 1:] - x_transform[:, :, :, :-1]))
            diff_y = torch.sum(torch.abs(x_transform[:, :, 1:, :] - x_transform[:, :, :-1, :]))
            tv_loss = tv_weight*(diff_x + diff_y)
            total_tv_loss += tv_loss.item()

            total_loss = style_loss + content_loss + tv_loss
            total_loss.backward()
            optimizer.step()

            if ((batch_num + 1) % 100 == 0):
            	print("{}  Epoch {}:  [{}/{}]  Batch:[{}]  Tol_style: {:.6f}  Tol_content: {:.6f}  Tol_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), ep + 1, img_count, len(train_dataset), batch_num+1,
                                total_style_loss/(batch_num+1.0), total_content_loss/(batch_num+1.0), total_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item())
            	     )

transformer.eval()
transformer.cpu()
if not os.path.exists("models"):
   os.makedirs("models")
filename = "models/MyStyle.pt"
torch.save(transformer.state_dict(), filename)
transformer.cuda()

