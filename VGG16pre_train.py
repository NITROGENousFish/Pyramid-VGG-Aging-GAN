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
cacd_pic_path = '../DATA/CACD/CACD_30_30_eyecroped'
data_dir = './data/DogsVSCats'
BATCH_SIZE = 16 
EPOCH_N = 2000

class CACD2K(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir    #存放人脸的数据位置
        self.transform = transform

    def __len__(self):
        return 163446

    def __getitem__(self, idx):
        filenames = os.listdir(self.root_dir)   #获取数据源列表名称
        filename = filenames[idx]   #获取图片文件名
        img = PILImage.open(os.path.join(self.root_dir, filename))
        if self.transform:
            img = self.transform(img)

        sample = {'image': img, 'age':self.ageclass(filename[0:2])}
        return sample

    def ageclass(self, age):
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

CACD_dataset = CACD2K(root_dir=cacd_pic_path,
                    transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                        ])
                    )

train_loader = DataLoader(dataset=CACD_dataset, 
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=True)
"""
    # 检查自己定义的数据集有啥问题
    for i in range(len(CACD_dataset)):
        sample = CACD_dataset[i]
        print(i, sample['image'].size(), sample['age'])
        if i == 3:
            break
"""

"""
    # 定义要对数据进行的处理
    data_transform = {x: transforms.Compose([transforms.Resize([224, 224]),
                                             transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                      for x in ["train", "valid"]}
    data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                             transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # 数据载入
    image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x), 
                                             transform=data_transform[x])
                     for x in ["train", "valid"]}
    # 数据装载
    dataloader = {x: torch.utils.data.DataLoader(dataset=image_datasets[x], 
                                                batch_size=16,
                                                shuffle=True)
                 for x in ["train", "valid"]}


    X_example, y_example = next(iter(dataloader["train"]))
    # print(u'X_example个数{}'.format(len(X_example)))
    # print(u'y_example个数{}'.format(len(y_example)))
    # print(X_example.shape)
    # print(y_example.shape)

    # 验证独热编码的对应关系
    index_classes = image_datasets["train"].class_to_idx
    # print(index_classes)
    # 使用example_classes存放原始标签的结果
    example_classes = image_datasets["train"].classes
    # print(example_classes)
"""

"""
    ### 图片预览
    img = torchvision.utils.make_grid(X_example)
    # print(img.shape)
    img = img.numpy().transpose([1, 2, 0])

    for i in range(len(y_example)):
        index = y_example[i]
        print(example_classes[index], end='   ')
        if (i+1)%8 == 0:
            print()

    # print(img.max())
    # print(img.min())
    # print(img.shape)

    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean

    # print(img.max())
    # print(img.min())
    # print(img.shape)

    plt.imshow(img)
    plt.show()
"""
    
# 下载已经具备最优参数的VGG16模型
model = models.vgg16(pretrained=True)
# 查看迁移模型细节
# print("迁移VGG16:\n", model)

# 对迁移模型进行调整
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
# 判断计算机的GPUs是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use_gpu = torch.cuda.is_available()
# if Use_gpu:
#     model = model.cuda()

# 定义代价函数和优化函数
loss_f = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)

# 模型训练和参数优化


if __name__ == "__main__":
    time_open = time.time()
    model = model.to(device)
    model.load_state_dict(torch.load("./vgg_age_classifier.pth"))
    for epoch in range(EPOCH_N):
        print("Epoch {}/{}".format(epoch+1, EPOCH_N))
        print("-"*10)
        model.train(True) # 设置为True，会进行Dropout并使用batch mean和batch 

        # if phase == "train":
        #     print("Training...")
        #     # 设置为True，会进行Dropout并使用batch mean和batch var
        #     model.train(True)
        # else:
        #     print("Validing...")    
        #      # 设置为False，不会进行Dropout并使用running mean和running var
        #     model.train(False)

        running_loss = 0.0
        running_corrects = 0

        # enuerate(),返回的是索引和元素值，数字1表明设置start=1，即索引值从1开始
        for batch, data in enumerate(train_loader, 1):
            # X: 图片，16*3*224*224; y: 标签，16
            X = data['image']  
            y = data['age']

            one_hot = torch.zeros(BATCH_SIZE, 6).scatter_(1, y.long().view(-1,1), 1).to(device)

            # print(X.size())    
            # print(y.size())
            X = X.to(device)
            y = y.to(device)
            print(X.size())
            y_pred = model(X)

            # pred，概率较大值对应的索引值，可看做预测结果
            _, pred = torch.max(y_pred.data, 1)

            # 梯度归零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_f(y_pred,one_hot)

            loss.backward()
            optimizer.step()

            # 计算损失和
            running_loss += float(loss)

            # 统计预测正确的图片数
            running_corrects += torch.sum(pred==y.data)

            # 共20000张测试图片，1250个batch，在使用500个及1000个batch对模型进行训练之后，输出训练结果
            print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4F}%".format(batch, running_loss/batch, 100*running_corrects/(BATCH_SIZE*batch)))
        
        epoch_loss = running_loss * BATCH_SIZE / len(CACD_dataset)
        epoch_acc = 100 * running_corrects / len(CACD_dataset)

        # 输出最终的结果
        print("Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))
        torch.save(model.state_dict(), 'vgg_age_classifier_'+str(epoch)+'.pth')
    # 输出模型训练、参数优化用时
    time_end = time.time() - time_open
    print("Total time: ",time_end)
