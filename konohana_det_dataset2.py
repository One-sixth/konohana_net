import numpy as np
import imageio
from my_py_lib.dataset.voc_dataset import VocDataset
import config.a_config as cfg
from my_py_lib import coord_tool
from time import time
import cv2
import my_prepro2


class KonohanaDataset:
    id_dir = 'sets'
    train_id_txt = 'train'
    # val_id_txt = 'val'
    image_dir = 'imgs'
    ann_dir = 'anns'
    voc_classes_name = cfg.all_classes+cfg.special_type


    def __init__(self, dataset_path='D:/Users/TWD/Desktop/konohana_dataset'):
        self.voc_dataset_dir = dataset_path
        self.voc_dataset = VocDataset(dataset_path, self.train_id_txt, self.id_dir, self.image_dir, self.ann_dir, self.voc_classes_name)

    def get_img_num(self):
        return self.voc_dataset.get_label_num()

    def __len__(self):
        return self.get_img_num()

    def get_batch_count(self, batch_size):
        return int(np.ceil(self.voc_dataset.get_label_num() / batch_size))

    def shuffle(self):
        self.voc_dataset.shuffle()

    def label_to_mat(self, predict_hw, img_hw, coords, classes, new_img_hw=None, exist_merge=True, keep_pixelunit=False,
                     use_special_type=True, ignore_unknow_label=False):
        '''
        转换 label 到矩阵
        :param predict_hw:
        :param img_hw:
        :param coords:
        :param classes:
        :param exist_merge: 对于重合框处理，True，取重叠框最大最小坐标，False，新的直接替换旧的
        :param keep_pixelunit: 是否保持包围框坐标为绝对坐标
        :param use_special_type: 是否使用 special 类
        :param ignore_unknow_label: 遇到未知标签，是则抛出异常，否则只输出信息
        :return:
        '''
        predict_hw = np.asarray(predict_hw, np.int32)

        label_confidence = np.zeros([*predict_hw, 1], np.float32)
        label_coords = np.zeros([*predict_hw, 4], np.float32)
        label_classes = np.zeros([*predict_hw, cfg.full_classes_len], np.float32)
        is_calc_loss = np.ones([*predict_hw, 1], np.float32)

        if new_img_hw is None:
            new_img_hw = img_hw

        if len(coords) != 0:
            # y1x1y2x2
            coords = coord_tool.xywh2yxhw(coords)
            # 转换为百分比坐标
            coords = coord_tool.coord_pixelunit_to_scale(coords, img_hw)
            # 得到包围框中心坐标
            center_yx = coord_tool.x1y1x2y2_to_xywh(coords)[:, :2]

            for classid, coord, center in zip(classes, coords, center_yx):
                classname = self.voc_classes_name[classid]
                if classname in cfg.all_classes:
                    # 得到格子坐标
                    cell_yx = np.asarray(center * predict_hw, np.int32)
                    cell_yx = np.clip(cell_yx, 0, predict_hw - 1)

                    if label_confidence[cell_yx[0], cell_yx[1], 0] > 0.:
                        if exist_merge:
                            # 如果该格子已经有东西了，将物体坐标转换为x1y1x2y2格式，取最大最小坐标，然后转回x_c,y_c,w,h坐标
                            coord1 = label_coords[cell_yx[0], cell_yx[1]]
                            coord[:2] = np.minimum(coord[:2], coord1[:2])
                            coord[2:] = np.maximum(coord[2:], coord1[2:])
                        else:
                            label_classes[cell_yx[0], cell_yx[1]] = np.zeros_like(label_classes[cell_yx[0], cell_yx[1]])
                    label_confidence[cell_yx[0], cell_yx[1], 0] = 1.
                    if keep_pixelunit:
                        coord = coord_tool.coord_scale_to_pixelunit(coord, new_img_hw)
                    label_coords[cell_yx[0], cell_yx[1]] = coord
                    label_classes[cell_yx[0], cell_yx[1], classid] = 1.

                elif classname in cfg.special_type and use_special_type:
                    # 目前只有不计算，不计算框框住区域，设置 is_calc_loss 为 0
                    coord = coord_tool.coord_scale_to_pixelunit(coord, predict_hw)
                    coord[:2] = np.clip(coord[:2], [0, 0], predict_hw - 1)
                    coord[2:] = np.clip(coord[2:], [0, 0], predict_hw - 1)
                    y1, x1, y2, x2 = np.asarray(coord, np.int32)
                    is_calc_loss[y1:y2, x1:x2, 0] = False
                else:
                    if ignore_unknow_label:
                        print('Unprocessed class label "%s" , but ignored' % classname)
                    else:
                        raise AttributeError('Unprocessed class label "%s"' % classname)
        label = np.concatenate([label_confidence, label_coords, label_classes, is_calc_loss], -1)
        return label

    def label_to_mat_fcos(self, predict_hw, img_hw, coords, classes, new_img_hw=None, exist_merge=True, keep_pixelunit=False,
                     use_special_type=True, ignore_unknow_label=False):
        '''
        转换 label 到矩阵
        :param predict_hw:
        :param img_hw:
        :param coords:
        :param classes:
        :param exist_merge: 对于重合框处理，True，取重叠框最大最小坐标，False，新的直接替换旧的
        :param keep_pixelunit: 是否保持包围框坐标为绝对坐标
        :param use_special_type: 是否使用 special 类
        :param ignore_unknow_label: 遇到未知标签，是则抛出异常，否则只输出信息
        :return:
        '''
        predict_hw = np.asarray(predict_hw, np.int32)

        label_confidence = np.zeros([*predict_hw, 1], np.float32)
        label_coords = np.zeros([*predict_hw, 4], np.float32)
        label_classes = np.zeros([*predict_hw, cfg.full_classes_len], np.float32)
        is_calc_loss = np.ones([*predict_hw, 1], np.float32)

        # # 冲突图
        # label_conflict = np.zeros([*predict_hw, 1], np.float32)

        # 最小面积图，用来处理最小优先问题
        label_area_size = np.full([*predict_hw, 1], np.inf, np.float32)

        calc_area_size = lambda y1x1y2x2: abs(y1x1y2x2[2]-y1x1y2x2[0]) * abs(y1x1y2x2[3]-y1x1y2x2[1])

        if new_img_hw is None:
            new_img_hw = img_hw

        if len(coords) != 0:
            # x1y1x2y2 转 y1x1y2x2
            coords = coord_tool.xywh2yxhw(coords)
            # 转换为百分比坐标
            coords = coord_tool.coord_pixelunit_to_scale(coords, img_hw)
            # # 得到包围框中心坐标，并转换单位为格子
            # center_yxhw = coord_tool.x1y1x2y2_to_xywh(coords)
            # # center_yx = coord_tool.coord_scale_to_pixelunit(center_yxhw, predict_hw)[:, :2]

            area_y1x1y2x2 = coord_tool.coord_scale_to_pixelunit(coords, predict_hw)
            # 限制坐标不会等于最大值，避免越界
            area_y1x1y2x2[:, 2:] = np.clip(area_y1x1y2x2[:, 2:], (0, 0), predict_hw - 1e-8)
            area_y1x1y2x2 = np.asarray(np.floor(area_y1x1y2x2), np.int)

            for classid, coord, area in zip(classes, coords, area_y1x1y2x2):

                classname = self.voc_classes_name[classid]
                if classname in cfg.all_classes:

                    if keep_pixelunit:
                        coord = coord_tool.coord_scale_to_pixelunit(coord, new_img_hw)
                    else:
                        raise AssertionError('keep_pixelunit must be true')

                    area_center = (abs(area[0] + area[2]) // 2, abs(area[1] + area[3]) // 2)

                    cur_area_size = calc_area_size(coord)

                    for y in range(area[0], area[2]+1):
                        for x in range(area[1], area[3]+1):

                            # 当前格子的中心，而不是左上角
                            c_y = y
                            c_x = x

                            t, l, b, r = (c_y - area[0], c_x - area[1], area[2] - c_y, area[3] - c_x)

                            # 当物体过小时，边界会非常小，此时就会出现以下情况，最好不要跳过，目前方法是强制设定为1
                            if area_center[0] == y and area_center[1] == x:
                                # 中心度，多数情况中心部位的值不是1，所以，手动设定设定处于正中心格子时，强制格子置信度为1
                                center_ness = 1.
                            else:
                                # 置信度，根据公式
                                center_ness = np.sqrt((min(t, b) / (max(t, b) + 1e-8)) * (min(l, r) / (max(l, r) + 1e-8)))
                                # center_ness = (min(t, b) / (max(t, b) + 1e-8)) * (min(l, r) / (max(l, r)))

                            if np.isnan(center_ness):
                                raise AssertionError('Found Nan')

                            # # 置信度设置为当前格子中心与物体真正中心的距离的开平方，开平方是为了减少置信度的衰减速度
                            # 这个无法归一化，弃用
                            # center_ness = np.sqrt(np.linalg.norm(center - current_cell_center))

                            # 这里应用越小越优先条件
                            # 就是优先选用小框
                            # 选用大框的条件是，该格子中心度比当前小框的中心度高于0.3时
                            # 选用小框的条件是，与上面相反

                            # 分类不互斥，一个像素可以属于多个类
                            label_classes[y, x, classid] = 1.

                            if label_confidence[y, x, 0] != 0.:
                                # 框内有物体
                                if label_area_size[y, x, 0] > cur_area_size:
                                    # 这是之前的框比当前的大
                                    if label_confidence[y, x, 0] - center_ness > 0.3:
                                        # 之前的框中心度比当前高，忽略
                                        continue
                                else:
                                    # 当前框比之前的大
                                    if center_ness - label_confidence[y, x, 0] < 0.3:
                                        # 之前的框中心度比当前高，忽略
                                        continue

                            # # 中心度代码
                            # if label_confidence[y, x, 0] > center_ness:
                            #     continue

                            # if label_confidence[y, x, 0] != 0:
                            #     # 如果之前此格子已经设置过物体，则清空
                            #     label_classes[y, x] = 0

                            label_confidence[y, x, 0] = center_ness

                            label_coords[y, x] = coord
                            # 分类提前到上面
                            # label_classes[y, x, classid] = 1.

                            # 这里存档当前应用框的面积
                            label_area_size[y, x, 0] = cur_area_size

                            # # 下面尝试使用边缘度，发现训练困难。。否决
                            # edge_ness = np.power(1-center_ness, 2)
                            # # 冲突检测和解决
                            # # 思想：累积冲突值，冲突值过高，无效该格子
                            # if label_conflict[y, x, 0] < 1.:
                            #     if label_confidence[y, x, 0] > edge_ness:
                            #         label_confidence[y, x, 0] -= edge_ness
                            #         label_conflict[y, x, 0] += edge_ness
                            #         continue
                            #     else:
                            #         edge_ness = edge_ness - label_confidence[y, x, 0]
                            #         label_conflict[y, x, 0] += label_confidence[y, x, 0]
                            # else:
                            #     edge_ness = 0
                            #
                            # if label_confidence[y, x, 0] != 0:
                            #     # 如果之前此格子已经设置过物体，则清空
                            #     label_classes[y, x] = 0
                            #
                            # label_confidence[y, x, 0] = edge_ness
                            #
                            # label_coords[y, x] = coord
                            # label_classes[y, x, classid] = 1.

                elif classname in cfg.special_type and use_special_type:
                    # 特殊分类目前只有不计算，不计算框框住区域，设置 is_calc_loss 为 0
                    coord = coord_tool.coord_scale_to_pixelunit(coord, predict_hw)
                    coord[:2] = np.clip(coord[:2], [0, 0], predict_hw - 1)
                    coord[2:] = np.clip(coord[2:], [0, 0], predict_hw - 1)
                    y1, x1, y2, x2 = np.asarray(coord, np.int32)
                    is_calc_loss[y1:y2, x1:x2, 0] = False
                else:
                    if ignore_unknow_label:
                        print('Unprocessed class label "%s" , but ignored' % classname)
                    else:
                        raise AttributeError('Unprocessed class label "%s"' % classname)
        label = np.concatenate([label_confidence, label_coords, label_classes, is_calc_loss], -1)
        return label

    def get_item(self, label_id, predict_hw, return_img_data=True, rescale_hw=None, exist_merge=True,
                 keep_pixelunit=True, use_special_type=True, ignore_unknow_label=False, *, like_fcos=False,
                 use_random_affine=False, use_augment_hsv=False, not_label=False):
        predict_hw = np.asarray(predict_hw, np.int32)

        label_info = self.voc_dataset.get_label_info(label_id)
        impath_or_img, width, height = label_info['image_path'], label_info['image_width'], label_info['image_height']
        classes, coords = self.voc_dataset.get_label_instance_bbox(label_id)
        # coords 格式为 x1y1x2y2

        if return_img_data:
            impath_or_img = imageio.imread(impath_or_img)

            if use_augment_hsv:
                my_prepro2.augment_hsv(impath_or_img)

            if use_random_affine:
                impath_or_img, coords, _ = my_prepro2.random_affine(impath_or_img, coords, borderValue=(0., 0., 0.))

            if rescale_hw:
                if np.all(np.array(rescale_hw) < np.array((height, width))):
                    impath_or_img = cv2.resize(impath_or_img, tuple(rescale_hw)[::-1], interpolation=cv2.INTER_AREA)
                else:
                    impath_or_img = cv2.resize(impath_or_img, tuple(rescale_hw)[::-1], interpolation=cv2.INTER_CUBIC)

        if not_label:
            return impath_or_img, np.array([0])

        if not like_fcos:
            label = self.label_to_mat(predict_hw, (height, width), coords, classes, new_img_hw=rescale_hw, ignore_unknow_label=ignore_unknow_label,
                                      use_special_type=use_special_type, exist_merge=exist_merge, keep_pixelunit=keep_pixelunit)
        else:
            label = self.label_to_mat_fcos(predict_hw, (height, width), coords, classes, new_img_hw=rescale_hw,
                                      ignore_unknow_label=ignore_unknow_label,
                                      use_special_type=use_special_type, exist_merge=exist_merge,
                                      keep_pixelunit=keep_pixelunit)
        return impath_or_img, label


    def get_batch(self, predict_hw, batch_id, batch_size, return_img_data=True, rescale_hw=None, exist_merge=True,
                 keep_pixelunit=True, use_special_type=True, ignore_unknow_label=False, *, like_fcos=False,
                  use_random_affine=False, use_augment_hsv=False):
        predict_hw = np.asarray(predict_hw, np.int32)

        batch_img = []
        batch_label = []

        for label_id in range(batch_id*batch_size, min((batch_id+1)*batch_size, self.voc_dataset.get_label_num())):
            impath_or_img, label = self.get_item(label_id, predict_hw, return_img_data=return_img_data,
                                                 rescale_hw=rescale_hw, exist_merge=exist_merge,
                                                 keep_pixelunit=keep_pixelunit, use_special_type=use_special_type,
                                                 ignore_unknow_label=ignore_unknow_label, like_fcos=like_fcos,
                                                 use_random_affine=use_random_affine, use_augment_hsv=use_augment_hsv)

            batch_img.append(impath_or_img)
            batch_label.append(label)

        if return_img_data:
            batch_img = np.asarray(batch_img)
        batch_label = np.asarray(batch_label)

        return batch_img, batch_label


if __name__ == '__main__':
    import eval_utils2

    khdataset = KonohanaDataset(r'D:\DeepLearningProject\datasets\konohana_dataset')
    bs = 2
    n_batch = khdataset.get_batch_count(bs)
    # imghw = (234, 416)
    # predict_hw = (15, 26)
    # predict_hw = (30, 52)
    # predict_hw = (234, 416)
    imghw = (360, 640)
    predict_hw = (45, 80)
    # imghw = (720, 1280)
    # predict_hw = (90, 160)
    # imghw = (540, 960)
    # predict_hw = (30, 52)

    cv2.resizeWindow('img', imghw[1], imghw[0])
    cv2.resizeWindow('center_ness', imghw[1], imghw[0])

    # import os
    # os.makedirs('dataset_out', exist_ok=True)

    for i in range(n_batch):
        t = time()
        # 测试数据集输出
        batch_img, batch_label = khdataset.get_batch(predict_hw, i, bs, True, imghw, like_fcos=True,
                                                     use_random_affine=True, use_augment_hsv=True)
        batch = np.split(batch_label, [1, 4+1, cfg.full_classes_len+4+1], -1)
        batch_center_ness = batch[0]
        batch_out = eval_utils2.get_bboxes_per_scale([batch[0], batch[0], batch[1], batch[2]], None, cfg.classes_mask, min_objectness=0.65, keep_classes_origin=True)
        batch_out = eval_utils2.nms_process(batch_out, 0.99)
        print(time()-t)
        for b in range(len(batch_img)):
            img = eval_utils2.draw_boxes_and_labels_to_image_multi_classes2(batch_img[b], batch_out[2][b], batch_out[0][b],
                                                                      batch_out[1][b], batch_out[3][b], cfg.classes_mask, 0.7,
                                                                      cfg.all_classes)
            # img = cv2.cvtColor(batch_img[b], cv2.COLOR_RGB2BGR)
            center_ness = batch_center_ness[b]
            center_ness = cv2.resize(center_ness, (imghw[1], imghw[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('center_ness', center_ness)

            confidence_ness = np.asarray((center_ness > 0.6) * 255, np.uint8)
            cv2.imshow('confidence_ness', confidence_ness)

            # imageio.imwrite('dataset_out/{}.jpg'.format(i*bs + b), img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            # cv2.waitKey(1)

        # # 测试fcos类型
        # img, label_mat = khdataset.get_item(i, predict_hw, True, imghw, True, True, True, False, like_fcos=True)
        # batch_center_ness = label_mat[..., :1]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', img)
        # cv2.imshow('batch_center_ness', batch_center_ness)
        # cv2.waitKey(0)
