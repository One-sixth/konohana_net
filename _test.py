from like_fcos.Detector import Detector
from konohana_det_dataset2 import KonohanaDataset
import config.a_config as cfg

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from progressbar import progressbar
import os
import eval_utils2

from model_utils_torch import *
import imageio
import cv2


# 9:16 比例
# img_hw = (234, 416)
img_hw = (360, 640)
# predict_hw = (15, 26)
predict_hw = (5, 5)

# 3G显存->16，8G显存->?
batch_size = 16
n_epoch = 1
konohana_dataset_path=r'D:/DeepLearningProject/datasets/konohana_dataset'
use_res2block = False


class KonohanaDatasetWarpper(Dataset):
    def __init__(self, predict_hw, img_hw):
        super().__init__()
        self.dataset = KonohanaDataset(konohana_dataset_path)
        self.predict_hw = predict_hw
        self.img_hw = img_hw

    def get_batch_count(self, batch_size):
        return self.dataset.get_batch_count(batch_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset.get_item(item, predict_hw=self.predict_hw, rescale_hw=self.img_hw, return_img_data=True,
                                     exist_merge=True, keep_pixelunit=True, use_special_type=True, ignore_unknow_label=False,
                                     like_fcos=True, not_label=True) #, use_random_affine=True, use_augment_hsv=True)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    out_dir = 'test_out'
    if use_res2block:
        out_dir = 'test_out_det3'
    os.makedirs(out_dir, exist_ok=True)

    khdataset = KonohanaDatasetWarpper(predict_hw, img_hw)
    loader = DataLoader(khdataset, batch_size, False, num_workers=3, timeout=10)
    net = Detector(img_hw, use_res2block=use_res2block).cuda()

    batch_count = khdataset.get_batch_count(batch_size)

    net_weight_path = 'net.pt'
    if use_res2block:
        net_weight_path = 'net_det3.pt'
    net.load_state_dict(torch.load(net_weight_path))

    print_params_size2(net)

    net.eval()

    for e in range(n_epoch):
        for b, (batch_img2, batch_label) in progressbar(enumerate(loader), prefix='epoch: {} '.format(e), max_value=batch_count):
            batch_img = (batch_img2.permute(0, 3, 1, 2).float() / (255 / 2) - 1).cuda()
            # batch_label = batch_label.permute(0, 3, 1, 2).cuda()

            net_out = net(batch_img)

            net_out = net_out.cpu().numpy()
            net_out = np.transpose(net_out, [0, 2, 3, 1])
            predict = np.split(net_out, [1, 2, 6], -1)

            # batch_out = eval_utils2.get_bboxes_per_scale_simple(predict, 0.7)
            batch_out = eval_utils2.get_bboxes_per_scale(predict, None, cfg.classes_mask, 0.6, True)
            # batch_out = [batch_out[0], batch_out[0], batch_out[2], batch_out[3]]
            batch_out = eval_utils2.nms_process(batch_out, 0.75)

            for b2 in range(len(batch_out[0])):
                ori_img = batch_img2[b2].numpy()
                img = eval_utils2.draw_boxes_and_labels_to_image_multi_classes2(ori_img, batch_out[2][b2],
                                                                                batch_out[0][b2],
                                                                                batch_out[1][b2], batch_out[3][b2],
                                                                                cfg.classes_mask, 0.1,
                                                                                cfg.classes)
                center_ness = net_out[b2, :, :, 0:1]
                center_ness = np.asarray(center_ness * 255, np.uint8)
                center_ness = cv2.resize(center_ness, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
                center_ness = center_ness[..., None]

                iou_scores = net_out[b2, :, :, 1:2]
                iou_scores = np.asarray(iou_scores * 255, np.uint8)
                iou_scores = cv2.resize(iou_scores, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_AREA)
                iou_scores = iou_scores[..., None]

                center_ness = np.tile(center_ness, [1, 1, 3])
                iou_scores = np.tile(iou_scores, [1, 1, 3])

                out_img1 = np.concatenate([ori_img, img], 1)
                out_img2 = np.concatenate([center_ness, iou_scores], 1)
                out_img = np.concatenate([out_img1, out_img2], 0)

                imageio.imwrite('{}/{}_{}.jpg'.format(out_dir, b, b2), out_img)

                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                cv2.imshow('out_img', out_img)
                cv2.waitKey(1)
