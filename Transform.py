import numpy as np
import torch
import os

from torch.autograd import Variable
from utils.imgprocess import img_transform_512,load_image,save_image
from net.transform import TransformNet

style_path = "ready2trans/content.jpg"
model_path = "models/NewMediaArt.pt"
#Lascaux.pt
#NewMediaArt.pt
dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content = load_image(style_path)
content = img_transform_512(content)
content = content.unsqueeze(0)
content = Variable(content).type(dtype)

model = TransformNet().type(dtype)
model.load_state_dict(torch.load(model_path))

stylized = model(content).cpu()
save_image("results/result.jpg", stylized.data[0])