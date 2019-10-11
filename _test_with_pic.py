"""
使用一个文件夹作为数据集

使用并行加速结构
"""

from like_fcos.Detector import Detector

import eval_utils2
import config.a_config as cfg

from model_utils_torch import *
import imageio
import cv2
import queue
import time
from threading import Thread
import os
from my_py_lib import im_tool


# 9:16 比例
img_hw = (360, 640)

# 3G显存->16，8G显存->?
batch_size = 16
n_epoch = 1
use_res2block = False
use_cuda = True

pic_dir_in = 'test_pic_in'
pic_dir_out = 'test_pic_out'

no_more = False
can_exit = False
write_cache = queue.Queue(200)
read_cache = queue.Queue(200)


def write_run():

    while True:
        try:
            b = write_cache.get(True, 5)
        except queue.Empty:
            time.sleep(1)
            if can_exit and write_cache.qsize() == 0:
                break
            continue

        batch_path, batch_img, net_out = b
        net_out = np.transpose(net_out, [0, 2, 3, 1])
        predict = np.split(net_out, [1, 2, 6], -1)

        batch_out = eval_utils2.get_bboxes_per_scale(predict, None, cfg.classes_mask, 0.65, True)
        batch_out = eval_utils2.nms_process(batch_out, 0.7)

        for b2 in range(len(batch_out[0])):
            ori_img = batch_img[b2]
            img = eval_utils2.draw_boxes_and_labels_to_image_multi_classes2(ori_img, batch_out[2][b2],
                                                                            batch_out[0][b2],
                                                                            batch_out[1][b2], batch_out[3][b2],
                                                                            cfg.classes_mask, 0.3,
                                                                            cfg.classes)
            center_ness = net_out[b2, :, :, 0:1]
            center_ness = np.asarray(center_ness * 255, np.uint8)
            center_ness = cv2.resize(center_ness, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
            center_ness = center_ness[:, :, None]

            iou_scores = net_out[b2, :, :, 1:2]
            iou_scores = np.asarray(iou_scores * 255, np.uint8)
            iou_scores = cv2.resize(iou_scores, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
            iou_scores = iou_scores[:, :, None]

            center_ness = np.tile(center_ness, [1, 1, 3])
            iou_scores = np.tile(iou_scores, [1, 1, 3])

            out_img1 = np.concatenate([ori_img, img], 1)
            out_img2 = np.concatenate([center_ness, iou_scores], 1)

            out_img = np.concatenate([out_img1, out_img2], 0)

            imageio.imwrite(os.path.join(pic_dir_out, batch_path[b2]), out_img)

            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('out_img', out_img)

            cv2.waitKey(16)


def read_run(pic_dir_in):
    global no_more
    batch_img = []
    batch_path = []

    for file in os.listdir(pic_dir_in):
        if os.path.splitext(file)[1] in ['.png', '.bmp', '.jpg']:
            im = imageio.imread(os.path.join(pic_dir_in, file))
            # im = cv2.resize(im, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
            im = im_tool.pad_picture(im, img_hw[1], img_hw[0], interpolation=cv2.INTER_AREA)
            batch_img.append(im)
            batch_path.append(file)


            if len(batch_img) > batch_size:
                read_cache.put([batch_path, np.asarray(batch_img, np.uint8)])
                # 注意不可用clear
                batch_img = []
                batch_path = []

    if len(batch_img) > 0:
        read_cache.put([batch_path, np.asarray(batch_img, np.uint8)])
    no_more = True


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    os.makedirs(pic_dir_in, exist_ok=True)
    os.makedirs(pic_dir_out, exist_ok=True)

    net = Detector(img_hw, use_res2block=use_res2block)
    net.eval()

    if use_cuda:
        net.cuda()

    net_weight_path = 'net.pt'
    if use_res2block:
        net_weight_path = 'net_det3.pt'
    net.load_state_dict(torch.load(net_weight_path))

    print_params_size2(net)

    # 使用 trace 加速
    # net = torch.jit.trace(net, torch.rand(batch_size, 3, *img_hw).cuda())

    run_thread = Thread(target=read_run, args=(pic_dir_in,))
    run_thread.start()
    write_thread = Thread(target=write_run)
    write_thread.start()

    while True:

        try:
            batch_path, batch_img = read_cache.get(True, 5)
        except queue.Empty:
            time.sleep(1)
            if no_more and read_cache.qsize() == 0:
                break
            continue

        batch_x = torch.from_numpy(batch_img)
        batch_x = (batch_x.permute(0, 3, 1, 2).float() / (255 / 2) - 1)

        if use_cuda:
            batch_x = batch_x.cuda()

        net_out = net(batch_x)

        net_out = net_out.cpu().numpy()

        batch_cache = [batch_path, batch_img, net_out]
        write_cache.put(batch_cache)

    can_exit = True
    write_thread.join()
