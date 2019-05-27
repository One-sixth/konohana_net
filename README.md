# konohana_net
这是模仿 mobilenet_v3 和 fcos 架构写的简易单端输出的目标检测网络。用于此花亭奇谭的目标识别。<br>
主要目的是用来玩。<br>
res2block 结构是模仿 res2net 里面的<br>
res2block 训练和预测需要的时间更长<br>
通过观察训练日志，可能是通道数较少和使用 dwconv 问题，res2block版本好像略差于普通版本<br>

## 依赖：
训练环境：GTX-1080-8G，GTX-970m-3G<br>
依赖：<br>
pytorch-1.1.0<br>
numpy<br>
progressbar2<br>
opencv-python<br>
imageio<br>
imageio-ffmpeg<br>
PIL<br>
matplotlib<br>

## 设计目标
1.支持高分辨率同时训练时显存占用必须在3G以内的（我的机器就3G显存）<br>
2.模型要简单而且便于修改<br>

## 关于预训练模型
模型可以在3G显存时进行训练，但模型并不是在我的电脑上训练的，训练时 batch_size 设定为 9，两张显卡各自负责一个模型的训练。<br>
1080ti-8g 平均2分50s每epoch<br>
970m-3g 平均9分10s每epoch<br>

常规模型并不是从0开始训练的，而是经过各种各样的修改，logs记录的是自从最后一次修改开始的。<br>
res2block的模型是从0开始训练的。<br>

## 测试
### 使用此花数据集进行测试
打开 _test.py 文件<br>
找到 konohana_dataset_path=xxx 变量<br>
将 xxx 改为此花数据集根目录路径<br>
如果要使用 res2net 版本，则需要修改为 use_res2block=True<br>
根据你的机器的显存大小适当设定 batch_size<br>
保存文件。<br>
执行 python3 _test.py<br>
已标记图像会输出到 test_out 文件夹内<br>
如果设定了 use_res2block=True，则会输出到 test_out_det3 中<br>

### 使用动漫视频文件进行测试
打开 _test_with_video.py 文件<br>
找到 video_in=xxx 变量<br>
将 xxx 改为输入的视频文件的路径<br>
找到 video_out=yyy 变量<br>
将 yyy 改为你要输出的视频名字<br>
如果要使用 res2net 版本，则需要修改为 use_res2block=True<br>
根据你的机器的显存大小适当设定 batch_size<br>
保存文件。<br>
执行 python3 _test_with_video.py 开始测试<br>
已标记的视频会输出到 video_out 指定的路径上<br>

### 此花数据集测试
test_out 中可以看到常规模型的输出<br>
test_out_det3 中可以看到res2block版模型输出<br>

![测试输出A](https://github.com/One-sixth/konohana_net/blob/master/test_out/1_0.jpg)
![测试输出B](https://github.com/One-sixth/konohana_net/blob/master/test_out/3_15.jpg)

### 动漫测试
等待补充

## 训练
net.pt optim.pt iter.txt 为普通版本模型的检查点文件<br>
net_det3.pt optim_det3.pt iter_det3.txt 为res2net版本模型的检查点文件<br>
如需重新训练则需要删除以上的检查点文件<br>

打开 _train.py 文件<br>
找到 konohana_dataset_path=xxx 变量<br>
将 xxx 改为此花数据集根目录路径<br>
如果要使用 res2net 版本，则需要修改为 use_res2block=True<br>
根据你的机器的显存大小适当设定 batch_size<br>
设定学习率 lr=1e-4，后面根据训练 epoch 数，手动变更学习率<br>
保存文件。<br>
执行 python3 _train.py.py 开始训练<br>
训练日志输出到 logs 文件夹中<br>
使用 start_tb.cmd 或者 start_tb.sh 启动 tensorboard 查看训练情况<br>

# 常规模型结构图<br>
![网络结构图](https://github.com/One-sixth/konohana_net/blob/master/net_struct.svg)
