# konohana_net

This is a object detection network that mimics the simple single-ended output written by the mobilenet_v3 and fcos architectures. Use of object detection with Konohana Kitan.<br>
The main purpose is to have fun.<br>
The res2block structure is mimicked inside res2net.<br>
Res2block training and forecasting takes longer.<br>
By observing the training log, it is possible that the number of channels is small and the dwconv is used. The res2block version seems to be slightly worse than the normal version.<br>

这是模仿 mobilenet_v3 和 fcos 架构写的简易单端输出的目标检测网络，用于此花亭奇谭的目标识别。<br>
主要目的是用来玩。<br>
res2block 结构是模仿 res2net 里面的。<br>
res2block 训练和预测需要的时间更长。<br>
通过观察训练日志，可能是通道数较少和使用 dwconv 问题，res2block版本好像略差于普通版本。<br>

## Dependent / 依赖：
Training environment<br>
训练环境：GTX-1080-8G，GTX-970m-3G<br>
Dependent<br>
依赖：<br>
pytorch-1.1.0<br>
numpy<br>
progressbar2<br>
opencv-python<br>
imageio<br>
imageio-ffmpeg<br>
PIL<br>
matplotlib<br>

## Design goals / 设计目标
1.Support high resolution and monitor memory usage must be within 3G (my machine is 3G memory)<br>
2.The model should be simple and easy to modify.<br>

1.支持高分辨率同时训练时显存占用必须在3G以内的（我的机器就3G显存）<br>
2.模型要简单而且便于修改<br>

## About the pre-training model / 关于预训练模型
The model can be trained in 3G memory, but the model is not trained on my computer. The batch_size is set to 9 during training.<br>
No multi-scale training is used.<br>
1080ti-8g average 2m50s per epoch.<br>
970m-3g average 9m10s per epoch.<br>

模型可以在3G显存时进行训练，但模型并不是在我的电脑上训练的，训练时 batch_size 设定为 9。<br>
没有使用多尺度训练。<br>
1080ti-8g 平均2分50s每epoch。<br>
970m-3g 平均9分10s每epoch。<br>

The regular model does not start training from 0, but through various modifications, the log records since the last modification. <br>
The res2block version model is trained from 0. <br>

常规模型并不是从0开始训练的，而是经过各种各样的修改，logs记录的是自从最后一次修改开始的。<br>
res2block的模型是从0开始训练的。<br>

The image size at the input of the model is 640x360, and the output is downsampled to 8x.<br>

模型输入端图像大小为 640x360，输出端下采样为8x.<br>

## Eval / 测试
### Test with konohana_dataset / 使用此花数据集进行测试
Open the _test.py file<br>
Find the konohana_dataset_path=xxx variable<br>
Change xxx to konohana_dataset root directory path<br>
If you want to use the res2block version, you need to change it to use_res2block=True<br>
Set batch_size according to the memory size of your machine<br>
save file. <br>
Execute python3 _test.py<br>
Marked images are output to the test_out folder<br>
If use_res2block=True is set, it will be output to test_out_det3 folder<br>

打开 _test.py 文件<br>
找到 konohana_dataset_path=xxx 变量<br>
将 xxx 改为此花数据集根目录路径<br>
如果要使用 res2block 版本，则需要修改为 use_res2block=True<br>
根据你的机器的显存大小适当设定 batch_size<br>
保存文件。<br>
执行 python3 _test.py<br>
已标记图像会输出到 test_out 文件夹内<br>
如果设定了 use_res2block=True，则会输出到 test_out_det3 文件夹内<br>

### Test with anime video files / 使用动漫视频文件进行测试
Open the _test_with_video.py file<br>
Find the video_in=xxx variable<br>
Change xxx to the path of the input video file<br>
Find the video_out=yyy variable<br>
Change yyy to the name of the video you want to output<br>
If you want to use the res2block version, you need to change it to use_res2block=True<br>
Set batch_size according to the memory size of your machine<br>
save file. <br>
Execute python3 _test_with_video.py to start the test<br>
The tagged video is output to the path specified by video_out<br>

打开 _test_with_video.py 文件<br>
找到 video_in=xxx 变量<br>
将 xxx 改为输入的视频文件的路径<br>
找到 video_out=yyy 变量<br>
将 yyy 改为你要输出的视频名字<br>
如果要使用 res2block 版本，则需要修改为 use_res2block=True<br>
根据你的机器的显存大小适当设定 batch_size<br>
保存文件。<br>
执行 python3 _test_with_video.py 开始测试<br>
已标记的视频会输出到 video_out 指定的路径上<br>

### Existing konohana_dataset test output / 现有的此花数据集测试输出
You can see more of the output of the regular model in test_out<br>
You can see the res2block version of the model output in test_out_det3<br>

test_out 中可以更多看到常规模型的输出<br>
test_out_det3 中可以更多看到res2block版模型输出<br>

![测试输出A](https://github.com/One-sixth/konohana_net/blob/master/test_out/1_0.jpg)
![测试输出B](https://github.com/One-sixth/konohana_net/blob/master/test_out/3_15.jpg)

### Existing animation test output / 动漫测试
Normal version model output
常规模型输出
https://github.com/One-sixth/konohana_net/blob/master/ep_10.mkv

res2block version model output
res2block版模型输出
https://github.com/One-sixth/konohana_net/blob/master/ep_10_det3.mkv

## Train / 训练
net.pt optim.pt iter.txt is checkpoint file for normal version model<br>
Net_det3.pt optim_det3.pt iter_det3.txt is checkpoint file for res2block version model<br>
If you need to retrain, you need to delete the above checkpoint file<br>

Open the _train.py file<br>
Find the konohana_dataset_path=xxx variable<br>
Change xxx to konohana_dataset root directory path<br>
If you want to use the res2block version, you need to change it to use_res2block=True<br>
Set batch_size according to the memory size of your machine<br>
Set the learning rate lr=1e-4, and manually change the learning rate according to the training epoch number<br>
Save file<br>
Start training with python3 _train.py.py<br>
Training log output to the logs folder<br>
Start tensorboard with start_tb.cmd or start_tb.sh to view training status<br>

net.pt optim.pt iter.txt 为普通版本模型的检查点文件<br>
net_det3.pt optim_det3.pt iter_det3.txt 为res2block版本模型的检查点文件<br>
如需重新训练则需要删除以上的检查点文件<br>

打开 _train.py 文件<br>
找到 konohana_dataset_path=xxx 变量<br>
将 xxx 改为此花数据集根目录路径<br>
如果要使用 res2block 版本，则需要修改为 use_res2block=True<br>
根据你的机器的显存大小适当设定 batch_size<br>
设定学习率 lr=1e-4，后面根据训练 epoch 数，手动变更学习率<br>
保存文件。<br>
执行 python3 _train.py.py 开始训练<br>
训练日志输出到 logs 文件夹中<br>
使用 start_tb.cmd 或者 start_tb.sh 启动 tensorboard 查看训练情况<br>

# 常规模型结构图<br>
![网络结构图](https://github.com/One-sixth/konohana_net/blob/master/net_struct.svg)
