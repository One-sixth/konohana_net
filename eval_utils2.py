import my_py_lib.coord_tool as coord_tool
import numpy as np
import my_py_lib.im_tool as im_tool
import cv2
from config import a_config as cfg


def _get_iou(box, boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 计算 bounding box 的长宽
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def _non_max_suppression(bboxes, iou_score, max_iou):
    ids = np.argsort(iou_score)[::-1]
    keep_box = []
    keep_box_id = []
    for i in ids:
        if len(keep_box) == 0:
            keep_box.append(bboxes[i])
            keep_box_id.append(i)
            continue
        keep_box_xywh = coord_tool.xywh_to_x1y1x2y2(keep_box)
        current_box = coord_tool.xywh_to_x1y1x2y2(bboxes[i])
        miou = np.max(_get_iou(current_box, keep_box_xywh))
        if miou > max_iou:
            continue
        keep_box.append(bboxes[i])
        keep_box_id.append(i)
    return keep_box_id


def get_bboxes_per_scale(predict, image_hw, classes_mask, min_objectness=0.5, keep_classes_origin=False):
    """
    筛选掉 confidence 小于 min_objectness 的框，并且将框格式从 yxhw 转为 x1y1x2y2，然后缩放为像素比例
    :param predict:
    :param image_hw: (height, weight) 或者 None，如果为None，将不对coords进行缩放处理
    :param classes_mask: 分类掩码
    :param min_objectness: 最小置信度，去掉置信度小于该值的框
    :param keep_classes_origin: True 则输出每个分类的分数，False 则输出类别编号
    :return:
    """
    confidence_batch, iou_score_batch, coords_batch, classes_batch = predict

    bs = confidence_batch.shape[0]
    confidence_batch = np.reshape(confidence_batch, (bs, -1))
    iou_score_batch = np.reshape(iou_score_batch, (bs, -1))
    coords_batch = np.reshape(coords_batch, (bs, -1, 4))
    classes_batch = np.reshape(classes_batch, (bs, -1, cfg.full_classes_len))

    coords_batch = coord_tool.yxhw2xywh(coords_batch)

    # 缩放 coords scale 到 pixelunit
    if image_hw is not None:
        coords_batch = coord_tool.coord_scale_to_pixelunit(coords_batch, image_hw)

    # 应用分类掩码
    classes_batch = classes_batch * np.reshape(classes_mask, [1, 1, -1])

    output_confidence_batch = []
    output_iou_score_batch = []
    output_coords_batch = []
    output_classes_batch = []

    for confidences, iou_scores, coords, classes  in zip(confidence_batch, iou_score_batch, coords_batch, classes_batch):
        select_bboxes = confidences > min_objectness
        if not np.any(select_bboxes):
            output_confidence_batch.append([])
            output_iou_score_batch.append([])
            output_coords_batch.append([])
            output_classes_batch.append([])
            continue

        confidences = confidences[select_bboxes]
        iou_scores  = iou_scores[select_bboxes]
        coords      = coords[select_bboxes]
        classes     = classes[select_bboxes]

        if not keep_classes_origin:
            classes = [np.argmax(cs) for cs in classes]

        output_confidence_batch.append(confidences)
        output_iou_score_batch.append(iou_scores)
        output_coords_batch.append(coords)
        output_classes_batch.append(classes)

    return output_confidence_batch, output_iou_score_batch, output_coords_batch, output_classes_batch


def get_bboxes_per_scale_simple(predict, min_objectness=0.5):
    """
    简化版，只需要置信度和坐标
    筛选掉 confidence 小于 min_objectness 的框，并且将框格式从 yxhw 转为 x1y1x2y2，然后缩放为像素比例
    :param net_out:
    :param image_hw: (height, weight) 或者 None，如果为None，将不对coords进行缩放处理
    :param classes_mask: 分类掩码
    :param min_objectness: 最小置信度，去掉置信度小于该值的框
    :param keep_classes_origin: True 则输出每个分类的分数，False 则输出类别编号
    :return:
    """
    confidence_batch, coords_batch = predict

    bs = confidence_batch.shape[0]
    confidence_batch = np.reshape(confidence_batch, (bs, -1))
    coords_batch = np.reshape(coords_batch, (bs, -1, 4))

    coords_batch = coord_tool.yxhw2xywh(coords_batch)

    output_confidence_batch = []
    output_coords_batch = []

    for confidences, coords in zip(confidence_batch, coords_batch):
        select_bboxes = confidences > min_objectness
        if not np.any(select_bboxes):
            output_confidence_batch.append([])
            output_coords_batch.append([])
            continue

        confidences = confidences[select_bboxes]
        coords      = coords[select_bboxes]

        output_confidence_batch.append(confidences)
        output_coords_batch.append(coords)

    return output_confidence_batch, output_coords_batch


def nms_process(predict, max_iou=0.7):
    '''
    NMS 过滤
    :param predict:
    :param max_iou:
    :return:
    '''
    confidence_batch, iou_score_batch, coords_batch, classes_batch = predict

    output_confidence = []
    output_iou_score = []
    output_coords = []
    output_classes = []

    for confidence, iou_score, coords, classes in zip(confidence_batch, iou_score_batch, coords_batch, classes_batch):
        select_bboxes = _non_max_suppression(coords, iou_score, max_iou)
        confidence = np.asarray(confidence, np.object)[select_bboxes]
        iou_score = np.asarray(iou_score, np.object)[select_bboxes]
        coords = np.asarray(coords, np.object)[select_bboxes]
        classes = np.asarray(classes, np.object)[select_bboxes]

        output_confidence.append(confidence)
        output_iou_score.append(iou_score)
        output_coords.append(coords)
        output_classes.append(classes)

    return output_confidence, output_iou_score, output_coords, output_classes


def nms_process_simple(predict, max_iou=0.7):
    '''
    NMS 过滤，简化版，只需要置信度和坐标
    :param predict:
    :param max_iou:
    :return:
    '''
    confidence_batch, coords_batch = predict

    output_confidence = []
    output_coords = []

    for confidence, coords in zip(confidence_batch, coords_batch):
        select_bboxes = _non_max_suppression(coords, confidence, max_iou)
        confidence = np.asarray(confidence, np.object)[select_bboxes]
        coords = np.asarray(coords, np.object)[select_bboxes]

        output_confidence.append(confidence)
        output_coords.append(coords)

    return output_confidence, output_coords


def draw_boxes_and_labels_to_image_multi_classes(image, classes, coords, scores=None, classes_name=None, classes_colors=None, font_color=[0, 0, 255]):
    """
    Draw bboxes and class labels on image. Return or save the image with bboxes
    Parameters
    -----------
    image : numpy.array
        The RGB image [height, width, channel].
    classes : list of int
        A list of class ID (int).
    coords : list of int
        A list of list for coordinates.
            - Should be [x, y, x2, y2]
    scores : list of float
        A list of score (float). (Optional)
    classes_name : list of str
        For converting ID to string on image.
    classes_colors : list of color
        A list of color [ [r,g,b], ...].
    font_color : front color
        Front color
    Returns
    -------
    numpy.array
        The output image.
    """
    image = image.copy()
    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 500)     # 粗细
    for i, _v in enumerate(coords):
        x, y, x2, y2 = np.asarray(coords[i], np.int32)
        bbox_color = [0, 255, 0] if classes_colors is None else classes_colors[classes[i]]
        cv2.rectangle(image, (x, y), (x2, y2), bbox_color, thick)
        if classes is not None:
            text = []
            for c in classes[i]:
                class_text = classes_name[c] if classes_name is not None else str(c)
                # score_text = " %.2f" % (scores[i]) if scores is not None else ''
                t = class_text #+ score_text
                text.append(t)
            text = '\n'.join(text)
            score_text = " %.2f" % (scores[i]) if scores is not None else ''
            text += score_text

            font_scale = 1.0e-3 * imh
            # text_size, _ = cv2.getTextSize(text, 0, font_scale, int(thick / 2) + 1)
            # cv2.rectangle(image, (x, y), (x+text_size[0], y-text_size[1]), bbox_color, -1)
            # cv2.putText(image, text, (x, y), 0, font_scale, font_color, int(thick / 3) + 1)
            image = im_tool.put_text(image, text, (x, y), font_scale*32, font_color, bbox_color)
    return image


def draw_boxes_and_labels_to_image_multi_classes2(image, coords, confidence_scores=None, iou_scores=None, classes=None, classes_mask=None,
                                                  classes_score_thresh=0.7, classes_name=None, classes_colors=None,
                                                  font_color=(0, 0, 255)):
    """
    Draw bboxes and class labels on image. Return or save the image with bboxes
    :param image: numpy.array
        The RGB image [height, width, channel].
    :param coords: list of int
        A list of list for coordinates.
            - Should be [x1, y1, x2, y2]
    :param confidence_scores: list of float
        A list of score (float). (Optional)
    :param iou_scores: list of float
        A list of score (float). (Optional)
    :param classes: classes scores matrix
        example [   [0.1,0.7,0.9],
                    [0.7,0.9,0.1],...]
    :param classes_score_thresh: float
        only show class when it scores greater than classes_score_thresh
    :param classes_name: list of str
        For converting ID to string on image.
    :param classes_colors: list of color
        A list of color [ [r,g,b], ...].
    :param font_color: front color
        Front color
    :return: numpy.array
        The output image.
    """
    image = image.copy()
    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 500)     # 粗细

    if classes is not None and classes_mask is not None:
        classes = np.asarray(classes)
        int_list = np.arange(len(classes_mask))

    for box_id, _ in enumerate(coords):
        x1, y1, x2, y2 = np.asarray(coords[box_id], np.int32)
        bbox_color = [0, 255, 0] if classes_colors is None else classes_colors[classes[box_id]]
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thick)
        text = []
        if confidence_scores is not None:
            conf_text = 'conf %.2f' % confidence_scores[box_id]
            text.append(conf_text)
        if iou_scores is not None:
            iou_text = 'iou %.2f' % iou_scores[box_id]
            text.append(iou_text)
        if classes is not None and classes_mask is not None:
            classes_ids = int_list[classes[box_id] * classes_mask > classes_score_thresh]
            classes_scores = classes[box_id][classes_ids]
            for classid, class_score in zip(classes_ids, classes_scores):
                class_text = classes_name[classid] if classes_name is not None else str(classid)
                class_text += " %.2f" % class_score
                text.append(class_text)

        if len(text) > 0:
            text = '\n'.join(text)
            font_scale = 1.0e-3 * imh
            # text_size, _ = cv2.getTextSize(text, 0, font_scale, int(thick / 2) + 1)
            # cv2.rectangle(image, (x, y), (x+text_size[0], y-text_size[1]), bbox_color, -1)
            # cv2.putText(image, text, (x, y), 0, font_scale, font_color, int(thick / 3) + 1)
            image = im_tool.put_text(image, text, (x1, y1), font_scale*32, font_color, bbox_color)
    return image


# def combind_multi_predict(predicts):
#     confidence_batch, iou_score_batch, coords_batch, classes_batch = predicts[0]
#
#     confidence_batch = [list(a) for a in confidence_batch]
#     iou_score_batch = [list(a) for a in iou_score_batch]
#     coords_batch = [list(a) for a in coords_batch]
#     classes_batch = [list(a) for a in classes_batch]
#
#     bs = len(confidence_batch)
#
#     for predict in predicts[1:]:
#         conf_bt, iou_bt, coord_bt, class_bt = predict
#         for i in range(bs):
#             confidence_batch[i].extend(list(conf_bt[i]))
#             iou_score_batch[i].extend(list(iou_bt[i]))
#             coords_batch[i].extend(list(coord_bt[i]))
#             classes_batch[i].extend(list(class_bt[i]))
#     return confidence_batch, iou_score_batch, coords_batch, classes_batch


# def y1x1x2y2_to_tlbr(coord_or_coords):
#     center = coord_tool.x1y1x2y2_to_xywh(coord_or_coords)[..., :2]
#     coord_or_coords[..., :2] = coord_or_coords[..., :2] - center
#     coord_or_coords[..., 2:] = coord_or_coords[..., 2:] - center
#     return coord_or_coords


if __name__ == '__main__':
    pass
