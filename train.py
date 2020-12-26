import torch
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime

from data_agu import Mydataset
from model import AttDinkNet34
from seg_iou import mean_IU
from loss import dice_bce_loss_with_logits

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.logging.set_verbosity(tf.logging.INFO)
pretrained_path = r'/home/vge/PycharmProjects/seg8/road/Result/11-29_08-55-28/24_checkpoint-best8358.pth'

class args:
    train_path = r'C:\Data\Road_Seg\data\data\train/train.csv'
    val_path = r'C:\Data\Road_Seg\data\data\val/test.csv'
    num_test_img = 4396

    result_dir = 'Result/'
    batch_size = 6
    learning_rate = 0.01
    max_epoch = 350

best_train_acc = 0.6
now_time = datetime.now()
time_str = datetime.strftime(now_time,'%m-%d_%H-%M-%S')
# 模型保存路径
log_dir = os.path.join(args.result_dir,time_str)
if not os.path.exists(log_dir):
     os.makedirs(log_dir)

writer = SummaryWriter(log_dir)
normMean = [0.4758, 0.4873, 0.5098, 0]
normStd = [0.1670, 0.1496, 0.1477, 1]
normTransfrom = transforms.Normalize(normMean, normStd)
transform = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom
    ])
# 数据加载，详见data_agu.py
train_data = Mydataset(path=args.train_path,transform=transform,augment=True)
val_data = Mydataset(path=args.val_path,transform=transform,augment=False)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size * 3, shuffle=False, drop_last=True, num_workers=2)

print("train data set:",len(train_loader)*args.batch_size)
print("valid data set:",len(val_loader))

net = AttDinkNet34(pretrained=True)
net.cuda()

if torch.cuda.is_available():
    # for continue training
    w = torch.Tensor([1.5, 1]).cuda()
    # continue training...
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load(pretrained_path)
    # net = net.to(device)
    # net.load_state_dict(checkpoint['state_dict'])
else:
    w = torch.Tensor([1.5, 1])

# 损失函数及优化方法定义
criterion4 = dice_bce_loss_with_logits().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, dampening=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True,min_lr=0.0000001)

# ---------------------------4、训练网络---------------------------
for epoch in range(args.max_epoch):
    loss_sigma = 0.0
    loss_val_sigma = 0.0
    acc_val_sigma = 0.0
    net.train()

    for i,data in enumerate(train_loader):
        inputs, labels,lab_name = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        labels = labels.float().cuda()
        optimizer.zero_grad()
        outputs = net.forward(inputs)
        # outputs=torch.sigmoid(outputs)
        outputs=torch.squeeze(outputs,dim=1)

        loss = criterion4(labels, outputs)
        loss.backward()
        optimizer.step()

        loss_sigma += loss.item()
        if i % 200 == 0 and i>0 :
            loss_avg = loss_sigma /200
            loss_sigma = 0.0
            tf.logging.info("Training:Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss:{:.4f}".format(
                epoch + 1, args.max_epoch,i+1,len(train_loader),loss_avg))
            writer.add_scalar("LOSS", loss_avg, epoch)

    # ---------------------------每个epoch验证网络---------------------------
    if epoch%1==0:
        net.eval()
        acc_val_sigma = 0
        acc_val = 0
        data_list = []
        for i, data in enumerate(val_loader):
            inputs, labels, img_name = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels = labels.float().cuda()
            with torch.no_grad():
                predicts = net.forward(inputs)

            predicts = torch.sigmoid(predicts)
            predicts[predicts < 0.5] = 0
            predicts[predicts >= 0.5] = 1
            result = np.squeeze(predicts)
            # outputs = torch.squeeze(outputs, dim=1)

            cc = labels.shape[0]
            for index in range(cc):
                # 评估方法为平均iou
                acc_val_sigma += mean_IU(labels[index].cpu().detach().numpy(), result[index].cpu().detach().numpy())

        # 验证精度提高时，保存模型
        val_acc = acc_val_sigma / args.num_test_img
        print("valid acc:", val_acc)
        print("lr:",args.learning_rate)
        print("best acc:", best_train_acc)
        scheduler.step(val_acc)
        if (val_acc) > best_train_acc:
            # state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            state = {'state_dict': net.state_dict()}
            filename = os.path.join(log_dir, str(epoch) + '_checkpoint-best.pth')
            torch.save(state, filename)
            best_train_acc = val_acc
            tf.logging.info('Save model successfully to "%s"!' % (log_dir + 'net_params.pkl'))
        tf.logging.info("After 1 epoch：acc_val:{:.4f},loss_val:{:.4f}".format(acc_val_sigma / (len(val_loader)),
                                                                              loss_val_sigma / (len(val_loader))))

writer.close()
net_save_path = os.path.join(log_dir,'net_params_end.pkl')
torch.save(net.state_dict(),net_save_path)