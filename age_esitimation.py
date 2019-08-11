import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt
import os 
import time
import PIL.Image as PILImage
import pdb
import openface
import cv2
PIC_PATH = "./me.jpg"
PTH_DIR = "./vgg_age_classifier.pth"
DLIB_PATH = "./shape_predictor_68_face_landmarks.dat"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(pth_dir_path):
    model = models.vgg16(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False

    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(4096, 4096),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(4096, 6),
                                      torch.nn.Sigmoid())
    # 查看调整后的迁移模型
    # print("调整后VGG16:\n", model)
    model.load_state_dict(torch.load(pth_dir_path))
    model = model.to(device)
    return model

def show_dlib_face(img_path,dlib_path):
    after_read = cv2.imread(img_path)
    openface_alilgn = openface.AlignDlib(dlib_path) #实例化openface的AlignDlib类
    return_rect = openface_alilgn.getLargestFaceBoundingBox(after_read) #返回dlib.rectanngle
    cropped = after_read[return_rect.left():return_rect.top(), return_rect.right():return_rect.bottom()]
    res = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA) #见下
    # cv2.rectangle(after_read,(return_rect.left(),return_rect.top()),(return_rect.right(),return_rect.bottom()),(0,255,0),2)
    # cv2.imshow("Picture",after_read)
    # cv2.waitKey(200000)
    return res
def ageclass(age):
    age = int(age)
    if age>=11 and age <=20:
        return 0
    elif age>=21 and age <=30:
        return 1
    elif age>=31 and age <=40:
        return 2
    elif age>=41 and age <=50:
        return 3
    elif age>=51 and age <=60:
        return 4
    else:
        return 5
if __name__ == "__main__":

    crop_pic = show_dlib_face(PIC_PATH,DLIB_PATH)
    temp = np.array(crop_pic).transpose((2,0,1))

    pic_torch = torch.from_numpy(temp).float().unsqueeze(0).to(device)
    model = load_model(PTH_DIR)
    model.train(False)
    predict = model(Variable(pic_torch))
    _, pred = torch.max(predict.data, 1)
    print(ageclass(25))
    print(predict)
    print(pred.data)