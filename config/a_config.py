import numpy as np
from collections import OrderedDict

full_classes_len = 30

classes = ['柚', '皋', '莲', '枣', '樱', '桐', '椿', '柊', '菖', '八百比丘尼', '药店老板', '瓜之介', '阿菊', '其他']
# classes_id = set(range(len(classes)))
n_classes = len(classes)

# 该类不计算分类loss
other_classes = []
other_classes_id = set(range(n_classes, n_classes+len(other_classes)))

all_classes = classes + other_classes
n_classes = len(all_classes)

# 该类包围框内的格子都不计算任何loss
special_type = ['不计算']

classes_mask = np.pad([True]*n_classes, [0, full_classes_len-n_classes], 'constant', constant_values=(False, False))
trust_iou_thresh=0.5
ignore_iou_thresh=0.9

max_item_num_each_cell=1


# anchor_boxes =\
# [
# 128, 207,
# 167, 167,
# 207, 128,
# 256, 414,
# 335, 335,
# 414, 256,
# 512, 828,
# 670, 670,
# 828, 512,
# ]
#
# anchor_boxes = np.asarray(np.reshape(anchor_boxes, [9, 2]), np.float32)


# classes_id_dict = dict(zip(all_classes, range(len(all_classes))))
# classes_id_dict = OrderedDict(sorted(classes_id_dict.items(), key=lambda t: t[1]))
