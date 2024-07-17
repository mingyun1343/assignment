# Assignment

## 目录
* [使用代码](#使用代码)
* [运行环境](#运行环境)
* [运行结果](#运行结果)
* [结果分析](#结果分析)

## 使用代码
上述文件中包含了所有训练数据和测试数据，也包含了使用默认参数训练的模型
### 直接使用模型
在根目录时，你可以直接使用以下代码使用该模型测试：
```
cd code
python train_convnet_pytorch.py
```
不出意外的话，你会得到以下或类似的结果：
```
learning_rate : 0.0001
max_steps : 5000
batch_size : 32
eval_freq : 500
data_dir : ./cifar10
load_model : True
Accuracy= 78.75
```
### 从头开始训练
在根目录时，你可以通过以下代码从头开始训练模型：
```
cd code
python train_convnet_pytorch.py --load_model False
```
这样你就会在model文件夹中得到你的模型，在训练结束后会自动使用模型测试一次。
## 运行环境
推荐的运行环境：
```
conda create -n ENV_NAME python=3.10
conda activate ENV_NAME
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
## 运行结果
### 默认参数运行
![默认参数运行](https://github.com/mingyun1343/assignment/raw/main/images/5000.png) 

### 调整参数运行
#### MAX_STEPS_DEFAULT=5000 && shuffle=True
![图片](https://github.com/mingyun1343/assignment/raw/main/images/5000+.png) 
#### MAX_STEPS_DEFAULT=10000 && shuffle=False
![图片](https://github.com/mingyun1343/assignment/raw/main/images/10000.png) 
#### MAX_STEPS_DEFAULT=10000 && shuffle=True
![图片](https://github.com/mingyun1343/assignment/raw/main/images/10000+.png) 
#### MAX_STEPS_DEFAULT=15000 && shuffle=False
![图片](https://github.com/mingyun1343/assignment/raw/main/images/15000.png) 
#### MAX_STEPS_DEFAULT=20000 && shuffle=False
![图片](https://github.com/mingyun1343/assignment/raw/main/images/20000.png) 
## 结果分析
在默认参数下，可以看出最终的准确度约为75%

可以看出，损失仍有明显下降的趋势，准确度仍有明显上升的趋势

因此，我调整MAX_STEPS_DEFAULT超参，加大训练轮次，以求获得更好的结果

在MAX_STEPS_DEFAULT=10000时，准确度确实有一定的提升，约为78%左右

但随着MAX_STEPS_DEFAULT=15000，准确度提升很少，约为79%左右，因此我认为再继续训练会造成过拟合等问题（甚至已经出现）

当MAX_STEPS_DEFAULT=20000，可以看出，准确度几乎没有上升，在79%波动，而损失函数却有一定程度的下降，可以认为出现了过拟合的现象

此外，我也尝试将训练时的shuffle=False改为shuffle=True

可以看出，虽然在准确度上的提升不是很大，但是两者的曲线对比可以看出，shuffle=True一组的曲线波动明显小，但波动更加频繁

个人认为shuffle=True更加优异，增加了训练时的随机性，提高其泛化能力
