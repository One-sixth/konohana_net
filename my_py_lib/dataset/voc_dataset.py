"""
This is a dataset for pascal voc 2012
"""

import os
from collections import OrderedDict
from xml.dom.minidom import parse
import imageio
import numpy as np

if __name__ == '__main__':
    from dataset import Dataset
else:
    from .dataset import Dataset


def _xml_extractor(xml_file):
    dom_tree = parse(xml_file)
    collection = dom_tree.documentElement
    file_name_xml = collection.getElementsByTagName('filename')[0]
    objects_xml = collection.getElementsByTagName('object')
    size_xml = collection.getElementsByTagName('size')

    file_name = file_name_xml.childNodes[0].data

    for size in size_xml:
        width = size.getElementsByTagName('width')[0]
        height = size.getElementsByTagName('height')[0]
        depth = size.getElementsByTagName('depth')[0]

        width = width.childNodes[0].data
        height = height.childNodes[0].data
        depth = depth.childNodes[0].data

    objects = []
    for object_xml in objects_xml:
        object_name = object_xml.getElementsByTagName('name')[0]
        bdbox = object_xml.getElementsByTagName('bndbox')[0]
        xmin = bdbox.getElementsByTagName('xmin')[0]
        ymin = bdbox.getElementsByTagName('ymin')[0]
        xmax = bdbox.getElementsByTagName('xmax')[0]
        ymax = bdbox.getElementsByTagName('ymax')[0]

        object = (object_name.childNodes[0].data, xmin.childNodes[0].data,
                  ymin.childNodes[0].data, xmax.childNodes[0].data,
                  ymax.childNodes[0].data)

        objects.append(object)

    return file_name, width, height, depth, objects


class VocDataset(Dataset):
    train_id_txt = 'ImageSets/Main/train.txt'
    val_id_txt = 'ImageSets/Main/val.txt'
    id_dir = 'ImageSets/Main'
    image_dir = 'JPEGImages'
    ann_dir = 'Annotations'

    classes_name = {'person': 0,
                    'bird': 1,
                    'cat': 2,
                    'cow': 3,
                    'dog': 4,
                    'horse': 5,
                    'sheep': 6,
                    'aeroplane': 7,
                    'bicycle': 8,
                    'boat': 9,
                    'bus': 10,
                    'car': 11,
                    'motorbike': 12,
                    'train': 13,
                    'bottle': 14,
                    'chair': 15,
                    'diningtable': 16,
                    'pottedplant': 17,
                    'sofa': 18,
                    'tvmonitor': 19
                    }

    def __init__(self, dataset_path, dataset_type, id_dir=None, image_dir=None, ann_dir=None, classes_name=None):
        """
        init voc dataset
        :param dataset_path:    dataset root dir path
        :param dataset_type:    filename in id_dir, for example: train.txt
        :param id_dir:          for custom voc dataset, where to search id file
        :param image_dir:       for custom voc dataset, where to search image file
        :param ann_dir:         for custom voc dataset, where to search ann file
        :param classes_name:    for custom voc dataset, classes name
        """
        Dataset.__init__(self)

        # for custom voc dataset
        self.id_dir = self.id_dir if id_dir is None else id_dir
        self.image_dir = self.image_dir if image_dir is None else image_dir
        self.ann_dir = self.ann_dir if ann_dir is None else ann_dir
        self.id_dir = self.id_dir if id_dir is None else id_dir
        if classes_name is not None:
            self.classes_name = dict(zip(classes_name, range(len(classes_name))))

        # become to orded dict
        self.classes_name = OrderedDict(sorted(self.classes_name.items(), key=lambda t: t[1]))

        # init basic info
        self.dataset_type = dataset_type
        self.dataset_dir = dataset_path

        # init path
        self.img_file_root = os.path.join(dataset_path, self.image_dir)
        self.label_file_root = os.path.join(dataset_path, self.ann_dir)
        self.label_file_list =\
            open(os.path.join(dataset_path, self.id_dir, dataset_type+'.txt'), 'r').read().splitlines()
        self.label_file_list_len = len(self.label_file_list)

        # init label info
        self.label_info = []
        for label_file in tuple(self.label_file_list):
            ann_file = os.path.join(self.label_file_root, label_file + '.xml')
            file_name, width, height, depth, objects = _xml_extractor(ann_file)
            classes = []
            coords = []
            for obj in objects:
                classname, xmin, ymin, xmax, ymax = obj
                classid = self.classes_name[classname]
                classes.append(classid)
                coords.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            classes = tuple(classes)
            coords = tuple(coords)
            self.label_info.append((file_name, int(width), int(height), int(depth), (classes, coords)))
        # self.label_info = tuple(self.label_info)


    # get the dataset information
    def get_label_num(self):
        """
        how many label in this dataset
        :return:
        """
        return self.label_file_list_len

    def get_class_num(self):
        """
        how many class in this dataset
        :return:
        """
        return len(self.classes_name.keys())

    def get_class_name(self):
        """
        get classes name
        :return:
        """
        return tuple(self.classes_name.keys())

    # Setting this dataset something
    def shuffle(self):
        """
        shuffle this dataset
        :return:
        """
        np.random.shuffle(self.label_info)

    # get item details
    def get_label_info(self, label_id):
        """
        get this label details, will return a dict, include key's
        [ image_path, image_width, image_height ]
        :param label_id:
        :return:
        """
        info = {}
        a = self.label_info[label_id]
        info['image_path'] = os.path.join(self.img_file_root, a[0])
        info['image_width'] = a[1]
        info['image_height'] = a[2]
        return info

    def get_label_image(self, label_id):
        """
        get origin image
        :param label_id: int
        :return: numpy.array
        """
        a = self.label_info[label_id]
        return imageio.imread(os.path.join(self.img_file_root, a[0]))

    def get_label_instance_bbox(self, label_id):
        """
        for object detection
        :param label_id:
        :return: a list of classes id, a list of coords (x1, y1, x2, y2)
        """
        a = self.label_info[label_id]
        return a[4] # (classes, coords)

    def get_label_class_mask(self, label_id):
        """
        for semantic segmentation
        :param label_id:
        :return:
        """
        self._not_imp()

    def get_label_instance_mask(self, label_id):
        """
        for instance segmentation
        :param label_id:
        :param class_id:
        :param instant_id:
        :return:
        """
        self._not_imp()


def test(voc_path='''E:\TDOWNLOAD\VOCdevkit\VOC2012'''):
    import sys
    sys.path.append('../')
    import im_tool
    ds = VocDataset(voc_path, 'train')
    ds.shuffle()
    print('image num', ds.get_label_num())
    print('class num', ds.get_class_num())
    print('class name', ds.get_class_name())

    label_id = np.random.randint(0, ds.get_label_num())
    print('label info', ds.get_label_info(label_id))

    classes_name = ds.get_class_name()
    print(ds.get_label_instance_bbox(label_id))
    classes, coords = ds.get_label_instance_bbox(label_id)
    for classid, coord in zip(classes, coords):
        print('name', classes_name[classid])
        print('bbox', coord)

    scores = np.ones_like(classes)

    draw_img = im_tool.draw_boxes_and_labels_to_image(ds.get_label_image(label_id), classes, coords, scores, classes_name)
    im_tool.show_image(draw_img)


if __name__ == '__main__':
    test()
