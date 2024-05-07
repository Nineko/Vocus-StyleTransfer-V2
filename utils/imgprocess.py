from PIL import Image
from torchvision import transforms
import numpy as np

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename)
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

# using ImageNet values
def normalize_tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


img_transform_512 = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    normalize_tensor_transform()
])

img_transform_256 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize_tensor_transform()
])
