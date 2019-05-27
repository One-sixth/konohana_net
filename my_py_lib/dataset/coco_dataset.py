"""
This is a dataset for coco 2017
"""

import os
from collections import OrderedDict
import imageio
import numpy as np
from pycocotools.coco import COCO

if __name__ == '__main__':
    from _cocostuffhelper import cocoSegmentationToSegmentationMap
    from dataset import Dataset
else:
    from ._cocostuffhelper import cocoSegmentationToSegmentationMap
    from .dataset import Dataset


class CocoDataset(Dataset):
    t_train2017 = 'train2017'
    t_val2017 = 'val2017'
    t2_instances = 'instances'
    t2_stuff = 'stuff'
    t2_person_keypoints = 'person_keypoints'

    def __init__(self, dataset_type, dataset_type2, dataset_path):
        Dataset.__init__(self)

        self.dataset_type = dataset_type
        self.dataset_type2 = dataset_type2
        self.dataset_path = dataset_path

        annFile_file = '%s/annotations/%s_%s.json' % (dataset_path, dataset_type2, dataset_type)
        self.cocoGt = COCO(annFile_file)
        self.img_list = self.cocoGt.getImgIds()

        categories = self.cocoGt.loadCats(self.cocoGt.getCatIds())
        self.classes_name = OrderedDict()
        # self.super_classes_name = OrderedDict()
        self.classes_id = OrderedDict()
        # self.super_classes_id = OrderedDict()
        self.coco_id_seq_id_map = {}
        self.keypoints_class_name = OrderedDict()
        self.keypoints_class_id = OrderedDict()
        self.keypoints_class_skeleton = []
        for i, cat in enumerate(categories):
            self.coco_id_seq_id_map[cat['id']] = i
            self.classes_name[cat['name']] = i
            self.classes_id[i] = cat['name']
            if 'keypoints' in dataset_type2:
                self.keypoints_class_name[cat['name']] = OrderedDict()
                self.keypoints_class_id[i] = OrderedDict()
                for k, n in enumerate(cat['keypoints']):
                    self.keypoints_class_name[cat['name']][n] = k
                    self.keypoints_class_id[i][k] = n
                self.keypoints_class_skeleton.append((np.asarray(cat['skeleton'])-1).tolist())


    # get the dataset information
    def get_label_num(self):
        """
        how many label in this dataset
        :return:
        """
        return len(self.img_list)

    def get_class_num(self):
        """
        how many class in this dataset
        :return:
        """
        return len(self.classes_name.keys())

    def get_keypoints_class_num(self, class_id_or_name):
        """
        how many keypoint class in this class
        :return:
        """
        if isinstance(class_id_or_name, str):
            return len(self.keypoints_class_name[class_id_or_name].keys())
        else:
            return len(self.keypoints_class_id[int(class_id_or_name)].keys())

    # def get_super_class_num(self):
    #     """
    #     only for coco dataset, how many super class in this dataset
    #     :return:
    #     """
    #     return len(self.super_classes_name.keys())

    def get_class_name(self):
        """
        get classes name
        :return:
        """
        return tuple(self.classes_name.keys())

    def get_keypoints_class_name(self, class_id_or_name):
        """
        get classes name
        :return:
        """
        if isinstance(class_id_or_name, str):
            return tuple(self.keypoints_class_name[class_id_or_name].keys())
        else:
            return tuple(self.keypoints_class_id[int(class_id_or_name)].values())

    def get_keypoints_class_skeleton(self, class_id_or_name):
        """
        get classes name
        :return:
        """
        if isinstance(class_id_or_name, str):
            return tuple(self.keypoints_class_skeleton[self.classes_name[class_id_or_name]])
        else:
            return tuple(self.keypoints_class_skeleton[int(class_id_or_name)])

    # def get_super_class_name(self):
    #     """
    #     get classes name
    #     :return:
    #     """
    #     return tuple(self.super_classes_name.keys())

    # Setting this dataset something
    def shuffle(self):
        """
        shuffle this dataset
        :return:
        """
        self._not_imp()

    # get item details
    def get_label_info(self, label_id):
        """
        get this label details, will return a dict, include key's
        [ image_path, image_width, image_height, image_depth ]
        :param label_id:
        :return:
        """
        info = {}
        imgId = self.img_list[label_id]
        img = self.cocoGt.imgs[imgId]
        info['image_path'] = os.path.join(self.dataset_path, self.dataset_type, img['file_name'])
        info['image_width'] = img['width']
        info['image_height'] = img['height']
        return info

    def get_label_image(self, label_id):
        """
        get origin image
        :param label_id: int
        :return: numpy.array
        """
        a = self.get_label_info(label_id)
        return np.asarray(imageio.imread(a['image_path']))

    def get_label_instance_bbox(self, label_id, *, iscrowd=None):
        """
        for object detection
        :param label_id:
        :return: a list of classes id, a list of coords (x1, y1, x2, y2)
        """
        imgId = self.img_list[label_id]
        annIds = self.cocoGt.getAnnIds(imgIds=imgId, iscrowd=iscrowd)
        anns = self.cocoGt.loadAnns(annIds)
        classes, coords = [], []
        for ann in anns:
            class_id = self.coco_id_seq_id_map[ann['category_id']]
            x, y, w, h = ann['bbox']
            x2, y2 = x+w, y+h
            classes.append(class_id)
            coords.append((int(x), int(y), int(x2), int(y2)))
        return tuple(classes), tuple(coords)

    def get_label_class_mask(self, label_id):
        """
        for semantic segmentation
        :param label_id:
        :return:
        """
        imgId = self.img_list[label_id]
        img_label = np.asarray(cocoSegmentationToSegmentationMap(self.cocoGt, imgId, includeCrowd=True), np.int)
        for x in range(len(img_label[0])):
            for y in range(len(img_label[1])):
                img_label[x][y] = self.coco_id_seq_id_map[img_label[x][y]]
        return img_label

    def get_label_instance_mask(self, label_id):
        """
        for instance segmentation
        :param label_id:
        :return:
        """
        imgId = self.img_list[label_id]
        annIds = self.cocoGt.getAnnIds(imgIds=imgId)
        anns = self.cocoGt.loadAnns(annIds)
        instance_masks = []
        for ann in anns:
            class_id = self.coco_id_seq_id_map[ann['category_id']]
            mask = self.cocoGt.annToMask(ann)
            instance_masks.append([class_id, mask])
        return instance_masks

    def get_label_instance_keypoints(self, label_id, iscrowd=None):
        """
        for person keypoints
        :param label_id:
        :return: [[x,y,v][x,y,v]...]
        """
        annIds = self.cocoGt.getAnnIds(imgIds=self.img_list[label_id], iscrowd=iscrowd)
        anns = self.cocoGt.loadAnns(annIds)
        keypoints = []
        for ann in anns:
            keypoints.append([self.coco_id_seq_id_map[ann['category_id']], np.reshape(ann['keypoints'], [-1, 3])])
        return keypoints


def test(dataset_type='train2017', dataset_type2=CocoDataset.t2_person_keypoints, dataset_root='E:\\TDOWNLOAD\\coco'):
    import sys
    sys.path.append('../')
    import im_tool

    ds = CocoDataset(dataset_type, dataset_type2, dataset_root)
    print('image num', ds.get_label_num())
    print('class num', ds.get_class_num())
    print('class name', ds.get_class_name())

    if dataset_type2 == CocoDataset.t2_person_keypoints:
        for c in range(ds.get_class_num()):
            print('keypoint main class name', ds.get_class_name()[c])
            print('keypoint class num', ds.get_keypoints_class_num(c))
            print('keypoint class name', ds.get_keypoints_class_name(c))
            print('keypoint class skeleton', ds.get_keypoints_class_skeleton(c))

    label_id = np.random.randint(0, ds.get_label_num())
    print('label info', ds.get_label_info(label_id))

    classes_name = ds.get_class_name()
    classes, coords = ds.get_label_instance_bbox(label_id)
    for classid, coord in zip(classes, coords):
        print('name', classes_name[classid])
        print('coord', coords)

    scores = np.ones_like(classes)
    image = ds.get_label_image(label_id)

    draw_img = im_tool.draw_boxes_and_labels_to_image(image, classes, coords, scores, classes_name)
    im_tool.show_image(draw_img)

    classes_colors = [im_tool.get_random_color() for _ in range(ds.get_class_num())]
    draw_img = im_tool.draw_boxes_and_labels_to_image(image, classes, coords, scores, classes_name, classes_colors)
    im_tool.show_image(draw_img)

    draw_img = im_tool.draw_boxes_and_labels_to_image(image, classes, coords, None, classes_name, classes_colors)
    im_tool.show_image(draw_img)

    if dataset_type2 == CocoDataset.t2_person_keypoints:
        draw_sk_img = image
        label_keypoints = ds.get_label_instance_keypoints(label_id)
        while len(label_keypoints) == 0:
            label_id = np.random.randint(0, ds.get_label_num())
            draw_sk_img = ds.get_label_image(label_id)
            label_keypoints = ds.get_label_instance_keypoints(label_id)
        for item in label_keypoints:
            keypoints_class_name = ds.get_keypoints_class_name(item[0])
            skelton = ds.get_keypoints_class_skeleton(item[0])
            keypoints = item[1]
            draw_sk_img = im_tool.draw_keypoints_and_labels_to_image_coco(draw_sk_img, keypoints, skelton, keypoints_class_name)
        im_tool.show_image(draw_sk_img)

    masks = ds.get_label_instance_mask(label_id)
    print(masks)


if __name__ == '__main__':
    test()
