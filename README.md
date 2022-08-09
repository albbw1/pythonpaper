基于DCL的CUB和leaf220分类

基本功能：实现数据集,CUB和leaf220分类任务

========================

环境依赖

pycharm ，python3.9.1 ，pytorch ，cuda10.2，  GPU 两块3080 

部署步骤

1. 安装pycharm，python，pytorch

2. 安装包 
pip install +(包)

torch，torch.nn ，optim， Variable， DataLoader，os，transforms, datasets, models
等

目录结构描述

			
|——datasets			//数据集
|——dcl-readme
|——models				
|——result_gather_net_models			
|——transform
|——utils
|——Github_commit.png
|——LICENSE.txt
|——README.md
|——all_acc.py
|——conda_list.txt
|——config.py
|——github__commit.png
|——test.py
|_____train.py


数据集

CUB:该数据集由加州理工学院在2010年提出的细粒度数据集，也是目前细粒度分类识别研究的基准图像数据集。
该数据集共有11788张鸟类图像，包含200类鸟类子类，每张图像均提供了图像类标记信息，图像中鸟的bounding box，鸟的关键part信息，以及鸟类的属性信息。

leaf220:未公开数据集，该数据集共有220类植物叶片。每类植物叶片图像大于30张。



简介
DCL提出一种“破坏和重建学习”方法来增强细粒度识别的难度并且训练分类模型来获取专家知识。
为了学到判别性区域和特征，除了标准的基础分类网络，该方法引入一个“破坏和重建”框架网络，破坏输入图像，然后重建输出入图像。
具体来说，对于“破坏”，DCL方法首先将输入图像划分为很多局部区域，然后通过区域混淆机制RCM来打乱，图像区域。
为了正确地识别这些破坏了的图像，分类网络必须将注意力更多的放在判别性的区域从而发现差异。为了补偿RCM引入的噪声，应用对抗性损失来区分破坏图像和原始图像，来抑制RCM引入的噪声模式。
为了重建图像，采用一种区域对齐网络来尝试恢复局部区域的原始空间布局，用于模拟局部区域之间的语义相关性。
通过训练时的参数共享，提出的DCL向分类网络中喂入了更多的判别性局部细节。本课程论文实现了DCL的官方数据集DCL方法复现，并将该方法引入自己数据集。实现分类任务。


实验步骤：
1.配置环境
2.复现基于CUB数据集的DCL方法。执行命令详见目录dcl-readme
3.在复现DCL方法后，未检查其泛化能力，将该方法用于leaf220数据集上，观察其是否过拟合。

值得注意一点 ,在config.py中将第11行注释掉，换成第10行注释掉的代码恢复。
数据集的设置，训练集，测试集，验证集已经设置好。若基于CUB数据集训练出错，
可将./datasets/CUB_200_2011/anno/路径中的文件ct_train.txt,ct_test.txt,ct_val.txt重命名为train.txt,test.txt,val.txt
训练阶段
在pycharm终端输入命令
python train.py --data CUB --epoch 360 --backbone resnet50 --tb 16 --tnw 0 --vb 32 --vnw 0 -- lr 0.0008 --lr_step 60 --cls_lr_ratio 10 --start_epoch 0 --detail training_descibe -- size 512 --crop 448 --cls_mul --swap_num 7 7
训练实践大约未两天。
测试阶段
在pycharm中端输入命令
python test.py --data CUB --backbone resnet50 --b 16 --nw 16 --ver val --crop 448 --swap_num 7 7
测试阶段需要加载训练阶段的权重，训练的的权重放在net_model中选择权重文件，将该文件转成pt,json文件
一并放入目录中的文件result_gather_net_model/training_descibe_52815_CUB中。在test.py文件中第93行更改
为resume=‘./net_model/ (该处修改为训练的权重路径)   ’第138行也更改对应的训练权重路径的json文件路径
结果显示为每一类分类正确率。

更换数据集为leaf220重复执行训练和测试 绘制损失函数曲线图观察是否过拟合（改代码已写好训练阶段结束会自动弹出）


