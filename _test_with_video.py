"""
使用视频做测试集

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


# 9:16 比例
img_hw = (360, 640)

# 3G显存->16，8G显存->?
batch_size = 16
n_epoch = 1
# konohana_dataset_path=r'D:\DeepLearningProject\datasets\konohana_dataset'
use_res2block = False
video_in = r"[ANK-Raws] このはな綺譚 01 (BDrip 1920x1080 HEVC-YUV420P10 FLAC).mkv"
video_out = 'out_ep_01.mkv'

can_exit = False
run_cache = queue.Queue(200)


def write_run():
    while True:
        try:
            b = run_cache.get(True, 5)
        except queue.Empty:
            time.sleep(1)
            if can_exit and run_cache.qsize() == 0:
                break
            continue

        batch_img, net_out = b
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

            video_out.append_data(out_img)

            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('out_img', out_img)
            cv2.waitKey(1000//24)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    video_dataset = imageio.get_reader(video_in)
    meta_data = video_dataset.get_meta_data()
    video_out = imageio.get_writer(video_out, codec=meta_data['codec'], fps=meta_data['fps'])

    net = Detector(img_hw, use_res2block=use_res2block).cuda()
    net.eval()

    net_weight_path = 'net.pt'
    if use_res2block:
        net_weight_path = 'net_det3.pt'
    net.load_state_dict(torch.load(net_weight_path))

    print_params_size2(net)

    write_thread = Thread(target=write_run)
    write_thread.start()

    while True:
        batch_img2 = []
        for _ in range(batch_size):
            try:
                in_img = video_dataset.get_next_data()
            except IndexError:
                break
            in_img = cv2.resize(in_img, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
            batch_img2.append(in_img)

        if len(batch_img2) == 0:
            break

        batch_img2 = np.asarray(batch_img2, np.uint8)
        batch_img = torch.tensor(batch_img2)
        batch_img = (batch_img.permute(0, 3, 1, 2).float() / (255 / 2) - 1).cuda()

        net_out = net(batch_img)

        net_out = net_out.cpu().numpy()

        batch_cache = [batch_img2, net_out]
        run_cache.put(batch_cache)


    can_exit = True
    write_thread.join()

    video_out.close()
