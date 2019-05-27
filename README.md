# konohana_net
这是模仿 mobilenet_v3 和 fcos 架构写的简易网络。用于此花亭奇谭的目标识别。
主要目的是用来玩。

## 依赖：
训练环境：GTX-1080-8G，GTX-970m-3G
依赖：
pytorch-1.1.0
numpy-1.16.3
progressbar2
opencv-python
imageio
imageio-ffmpeg
PIL
matplotlib


## 设计目标
1.支持高分辨率同时训练时显存占用必须在3G以内的（我的机器就3G显存）
2.模型要简单而且便于修改


## 关于预训练模型
模型可以在3G显存时进行训练，但模型并不是在我的电脑上训练的，因为太慢了，平均5分钟 1 epoch，1080-8g的训练速度是970m-3g训练速度的4.5倍，是在学校的深度学习服务器，配置为双卡GTX-1080-8G上训练的，训练时 batch_size 设定为 9，两张显卡各自负责一个模型的训练。

常规模型并不是从0开始训练的，而是经过各种各样的修改，logs记录的是自从最后一次修改开始的。
res2block的模型是从0开始训练的。


## 测试
### 使用此花数据集进行测试
打开 _test.py 文件
找到 konohana_dataset_path=xxx 变量
将 xxx 改为此花数据集根目录路径
如果要使用 res2net 版本，则需要修改为 use_res2block=True
根据你的机器的显存大小适当设定 batch_size
保存文件。
执行 python3 _test.py
已标记图像会输出到 test_out 文件夹内
如果设定了 use_res2block=True，则会输出到 test_out_det3 中

### 使用动漫视频文件进行测试
打开 _test_with_video.py 文件
找到 video_in=xxx 变量
将 xxx 改为输入的视频文件的路径
找到 video_out=yyy 变量
将 yyy 改为你要输出的视频名字
如果要使用 res2net 版本，则需要修改为 use_res2block=True
根据你的机器的显存大小适当设定 batch_size
保存文件。
执行 python3 _test_with_video.py 开始测试
已标记的视频会输出到 video_out 指定的路径上

### 此花数据集测试
等待补充

### 动漫测试
等待补充

## 训练
net.pt optim.pt iter.txt 为普通版本模型的检查点文件
net_det3.pt optim_det3.pt iter_det3.txt 为res2net版本模型的检查点文件
如需重新训练则需要删除以上的检查点文件

打开 _train.py 文件
找到 konohana_dataset_path=xxx 变量
将 xxx 改为此花数据集根目录路径
如果要使用 res2net 版本，则需要修改为 use_res2block=True
根据你的机器的显存大小适当设定 batch_size
设定学习率 lr=1e-4，后面根据训练 epoch 数，手动变更学习率
保存文件。
执行 python3 _train.py.py 开始训练
训练日志输出到 logs 文件夹中
使用 start_tb.cmd 或者 start_tb.sh 启动 tensorboard 查看训练情况


# 常规模型结构图<br>
等待补充
