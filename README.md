# Assignment

## 目录
* 使用代码
* 运行环境
* 运行结果

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
#### 结果分析

