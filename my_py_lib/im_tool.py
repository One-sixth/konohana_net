# 图像处理

import imageio
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont


def show_image(img):
    """
    show image
    :param img: numpy array
    :return: None
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('show_img', img)
    cv2.waitKey(0)


def pad_picture(img, width, height):
    """
    padded picture to specified shape, then return this and padded mask
    :param img: input numpy array
    :param width: output image width
    :param height: output image height
    :return: output numpy array
    """
    s_height, s_width, s_depth = img.shape
    # ratio = s_width / s_height
    width_prop = width / s_width
    height_prop = height / s_height
    min_prop = min(width_prop, height_prop)
    img = cv2.resize(img, (int(s_width * min_prop), int(s_height * min_prop)), interpolation=cv2.INTER_NEAREST)
    img_start_x = width / 2 - s_width * min_prop / 2
    img_start_y = height / 2 - s_height * min_prop / 2
    new_img = np.zeros((height, width, s_depth), np.float32)
    new_img[int(img_start_y):int(img_start_y)+img.shape[0], int(img_start_x):int(img_start_x)+img.shape[1]] = img
    return new_img


def crop_picture(img, width, height):
    """
    padded picture to specified shape
    :param img: input numpy array
    :param width: output image width
    :param height: output image height
    :return: output numpy array
    """
    s_height, s_width, s_depth = img.shape
    s_ratio = s_width / s_height
    ratio = width / height
    if s_ratio > ratio:
        need_crop_x = (s_width - s_height * ratio) / 2
        new_width = s_height * ratio
        img = img[0:s_height, int(need_crop_x):int(need_crop_x+new_width), :]
    elif s_ratio < ratio:
        need_crop_y = (s_height - s_width / ratio) / 2
        new_height = s_width / ratio
        img = img[int(need_crop_y):int(need_crop_y+new_height), 0:s_width, :]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


_default_font_path = os.path.join(os.path.dirname(__file__), 'SourceHanSansCN-Regular.otf')

def put_text(img, text, pos, font_size=20, font_color=(0, 0, 255), bg_color=None, font_type=None):
    """
    draw font on image
    :param img: np.array
        input image
    :param text: str
        the text will draw on image
    :param pos: (x, y)
        where to draw text
    :param font_size: int
        like the name
    :param font_color: (r,g,b)
        0-255, like the name
    :param font_type: str
        which font would you want
    :return: np.array
        output image
    """
    if bg_color is not None and bg_color is not str:
        bg_color = tuple(bg_color)

    if font_type is None:
        font_type = _default_font_path

    pil_im = Image.fromarray(np.asarray(img, np.uint8))
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(font_type, int(font_size))
    w, h = draw.multiline_textsize(text, font)
    draw.rectangle((pos[0], pos[1], pos[0]+w, pos[1]+h), bg_color)
    draw.multiline_text(pos, text, tuple(font_color), font=font)
    return np.asarray(pil_im, np.uint8)


_start_color = np.array([64, 128, 192])
_color_step = np.array([173, 79, 133])

def get_random_color():
    """
    Get random color
    :return: np.array([r,g,b])
    """
    global _start_color, _color_step
    # rgb = np.random.uniform(0, 25, [3])
    # rgb = np.asarray(np.floor(rgb) / 24 * 255, np.uint8)
    _start_color = (_start_color + _color_step) % np.array([256, 256, 256])
    rgb = np.asarray(_start_color, np.uint8).tolist()
    return rgb


def draw_keypoints_and_labels_to_image_coco(image, keypoints, skeleton, keypoints_name_list):
    """
    draw keypoints with coco data type
    :param image:
    :param keypoints:
    :param skeleton:
    :param keypoints_name_list:
    :return:
    """
    image = image.copy()
    imh, imw = image.shape[0:2]
    thick = int((imh + imw) // 500)
    for k, kpt in enumerate(keypoints):
        kpt = np.asarray(kpt, np.int32)
        x, y, v = kpt
        if kpt[2] == 0:
            continue
        cv2.circle(image, (x, y), max(int(1.5e-3 * imh), 1), (0, 0, 255), -1)
        text = keypoints_name_list[k]
        cv2.putText(image, text, (x, y), 0, 1.5e-3 * imh, [0, 0, 255], int(thick / 3) + 1)
    for s in skeleton:
        x, y, v = keypoints[s[0]]
        x2, y2, v2 = keypoints[s[1]]
        if v > 0 and v2 > 0:
            cv2.line(image, (x, y), (x2, y2), [255, 0, 0], thick)
    return image


def draw_boxes_and_labels_to_image(image, classes, coords, scores=None, classes_list=None, classes_colors=None, font_color=[0, 0, 255]):
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
    classes_list : list of str
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
            class_text = classes_list[classes[i]] if classes_list is not None else str(classes[i])
            score_text = " %.2f" % (scores[i]) if scores is not None else ''
            text = class_text + score_text

            font_scale = 1.0e-3 * imh
            # text_size, _ = cv2.getTextSize(text, 0, font_scale, int(thick / 2) + 1)
            # cv2.rectangle(image, (x, y), (x+text_size[0], y-text_size[1]), bbox_color, -1)
            # cv2.putText(image, text, (x, y), 0, font_scale, font_color, int(thick / 3) + 1)
            image = put_text(image, text, (x, y), font_scale*32, font_color, bbox_color)
    return image


def test():
    mod_dir = os.path.dirname(__file__)

    im = imageio.imread(mod_dir + '/laska.png')

    # test pad_picture
    new_img = pad_picture(im, 1024, 700)
    show_image(np.asarray(new_img, np.uint8))

    new_img = pad_picture(im, 700, 1024)
    show_image(np.asarray(new_img, np.uint8))

    # test crop_picture
    new_img = crop_picture(im, 1024, 512)
    show_image(np.asarray(new_img, np.uint8))

    new_img = crop_picture(new_img, 512, 1024)
    show_image(np.asarray(new_img, np.uint8))

    # test put_text
    new_img = pad_picture(im, 512, 512)
    new_img = put_text(new_img, '你好world', (256, 256))
    show_image(np.asarray(new_img, np.uint8))

    # test get_random_color
    m = np.zeros([16 * 16, 3], np.uint8)
    for i in range(16 * 16):
        m[i] = get_random_color()
    m = np.reshape(m, (16, 16, 3))
    m = cv2.resize(m, (256, 256), interpolation=cv2.INTER_NEAREST)
    # m = np.asarray(resize(m, (256, 256, 3), 0, 'constant', preserve_range=True, anti_aliasing=False), np.uint8)
    show_image(m)


if __name__ == '__main__':
    test()
