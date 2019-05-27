import config.a_config as cfg
from like_fcos.Detector import Detector
from like_fcos.det_loss import DetLoss
from konohana_det_dataset2 import KonohanaDataset

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from progressbar import progressbar
import os

from model_utils_torch import *

# 9:16 比例
img_hw = (360, 640)
predict_hw = (45, 80)

# 学习率，1-50，1e-4；50-100，1e-5；100-350，1e-6；350-1000，1e-7
lr = 1e-6
# 3G显存->3，8G显存->9
batch_size = 3
n_epoch = 1000
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
                                     like_fcos=True, use_random_affine=True, use_augment_hsv=True)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    classes_mask = torch.tensor(cfg.classes_mask, dtype=torch.float32).cuda()

    khdataset = KonohanaDatasetWarpper(predict_hw, img_hw)
    loader = DataLoader(khdataset, batch_size, True, num_workers=4, timeout=10)
    net = Detector(img_hw, use_res2block=use_res2block).cuda()
    losser = DetLoss()

    batch_count = khdataset.get_batch_count(batch_size)

    optimer = optim.Adam(net.parameters(), lr, (0.5, 0.999), weight_decay=lr*1e-2)

    start_epoch = 0

    net_weight_path = 'net.pt'
    optimer_weight_path = 'optim.pt'
    iter_txt = 'iter.txt'
    if use_res2block:
        net_weight_path = 'net_det3.pt'
        optimer_weight_path = 'optim_det3.pt'
        iter_txt = 'iter_det3.txt'

    if os.path.isfile(net_weight_path):
        net.load_state_dict(torch.load(net_weight_path))
        optimer.load_state_dict(torch.load(optimer_weight_path))
        if os.path.isfile(iter_txt):
            start_epoch = int(open(iter_txt, 'r').read(10))

    print_params_size2(net)

    net.train()
    for e in range(start_epoch, n_epoch):
        khdataset.dataset.shuffle()
        for b, (batch_img, batch_label) in progressbar(enumerate(loader), prefix='epoch: {} '.format(e), max_value=batch_count):
            batch_img = (batch_img.permute(0, 3, 1, 2).float() / (255 / 2) - 1).cuda()
            batch_label = batch_label.permute(0, 3, 1, 2).cuda()

            net_out = net(batch_img)
            loss = losser(net_out, batch_label, classes_mask, global_step=e * batch_count + b)
            optimer.zero_grad()
            loss.backward()
            if np.isnan(loss.item()):
                raise AssertionError('Found Nan')
            # 梯度裁剪
            nn.utils.clip_grad_norm_(net.parameters(), 2)
            optimer.step()

            if (b+1) % 100 == 0:
                torch.save(net.state_dict(), net_weight_path)
                torch.save(optimer.state_dict(), optimer_weight_path)
                open(iter_txt, 'w').write(str(e))
        torch.save(net.state_dict(), net_weight_path)
        torch.save(optimer.state_dict(), optimer_weight_path)
        open(iter_txt, 'w').write(str(e + 1))