# 2020“华为云杯”人工智能大赛冠军方案分享
### 我们是西南交通大学朱庆教授带领的虚拟地理环境团队，欢迎访问我们的主页：https://vrlab.org.cn/ 了解更多信息。我们的方案在初赛的线上精度达0.8411，排名（2/377）

## 总体方案介绍：
##### 针对遥感影像中道路尺度差异大，道路与其它背景信息样本不平衡，传感器、环境、构筑材料差异导致外观多样化等特点。我们在E-D架构的基础上，提出一种通道注意力增强的特征自适应融合方法，并设计基于梯度的边缘约束模块。在增强空间细节和语义特征的同时，提高道路边缘的特征响应，实现多尺度道路准确提取
![Alt text](https://github.com/liaochengcsu/road_segmentation_pytorch/blob/main/net.jpg)

## 方案策略总结：
 * 利用随机平衡采样法采样影像；
 * 训练过程中使用随机翻转，旋转，缩放，颜色空间变换等数据增强方法；
 * 使用BCE+Dice做损失函数，有助于类别不平衡样本的模型优化；
 * 引入基于梯度的边缘特征提取模块，使提取的道路边缘更精细；
 * 引入空间和通道注意力机制，分别提高道路完整性和特征自适应融合；
 * 使用ImageNet预训练的ResNext200做预训练；
 * BCE损失函数类别加权；
 * 预测过程使用原始+水平翻转+180旋转的平均值；
 * 对预测结果做空洞填充和噪声去除。
![Alt text](https://github.com/liaochengcsu/road_segmentation_pytorch/blob/main/data-agu.jpg)
## 训练自定义数据：
##### 1. 数据集准备。将原始影像和标签进行随机采样，裁剪后的标签文件名与影像相同，分别保存在两个文件夹下。将文件名写入csv文件，如下图所示：
![Alt text](https://github.com/liaochengcsu/road_segmentation_pytorch/blob/main/data_tree.jpg)
##### 2. train.py中修改第25-27行的csv文件路径及验证集数量
```
train_path = r'C:\Data\Road_Seg\data\data\train/train.csv'
val_path = r'C:\Data\Road_Seg\data\data\val/.csv'
num_test_img = 4396
```
##### 3. data_agu.py文件第122-123、144-145行读取csv并获取文件绝对路径
```
fn = os.path.join(self.file_path, "images/" + fn)
label = os.path.join(self.file_path, "labels/" + lab)
```
##### 4. 程序训练过程中自动下载基于ImageNet的预训练模型，执行完一个epoch后会计算验证集精度，训练完成后将模型按照初赛格式提交即可。我们的验证结果如下图：
![Alt text](https://github.com/liaochengcsu/road_segmentation_pytorch/blob/main/result.jpg)
