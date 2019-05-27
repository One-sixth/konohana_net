import numpy as np


def xywh2yxhw(coord_or_coords):
    coord_or_coords = np.asarray(coord_or_coords)
    yx = coord_or_coords[..., :2][..., ::-1]
    hw = coord_or_coords[..., 2:][..., ::-1]
    yxhw = np.concatenate([yx, hw], -1)
    return yxhw


def yxhw2xywh(coord_or_coords):
    return xywh2yxhw(coord_or_coords)


def coord_pixelunit_to_scale(coord_or_coords, shape):
    """
    Scale down a list of coordinates from pixel unit to the ratio of image size i.e. in the range of [0, 1].
    Parameters
    ------------
    coords : list of list of 4 ints or None
        For coordinates of more than one images .e.g.[[x, y, w, h], [x, y, w, h], ...].
    shape : list of 2 int or None
        [width, height], when coords is xywh, x1y1x2y2 etc
        [height, width]  when coords is yxhw, y1x1y2x2 etc
    Returns
    -------
    list of list of 4 numbers
        A list of new bounding boxes.
    """
    coord_or_coords = np.asarray(coord_or_coords)
    shape = [*shape, *shape]
    return coord_or_coords / shape


def coord_scale_to_pixelunit(coord_or_coords, shape):
    """
    Convert one coordinate [x, y, w (or x2), h (or y2)] in ratio format to image coordinate format.
    It is the reverse process of ``coord_pixelunit_to_scale``.
    Parameters
    -----------
    coord : list of 4 float
        One coordinate of one image [x, y, w (or x2), h (or y2)] in ratio format, i.e value range [0~1].
    shape : tuple of 2 or None
        For [height, width].
    Returns
    -------
    list of 4 numbers
        New bounding box.

    """
    coord_or_coords = np.asarray(coord_or_coords)
    shape = [*shape, *shape]
    return coord_or_coords * shape


def xywh_to_x1y1x2y2(coord_or_coords):
    """
    Convert one coordinate [x_center, y_center, w, h] to [x1, y1, x2, y2] in up-left and botton-right format.
    Parameters
    ------------
    coord : list of 4 int/float
        One or multi coordinate.
    Returns
    -------
    list of 4 numbers
        New bounding box.
    """
    coord_or_coords = np.asarray(coord_or_coords)
    x1y1 = coord_or_coords[..., :2] - coord_or_coords[..., 2:] / 2
    x2y2 = coord_or_coords[..., :2] + coord_or_coords[..., 2:] / 2
    x1y1x2y2 = np.concatenate([x1y1, x2y2], -1)
    return x1y1x2y2


def x1y1x2y2_to_xywh(coord_or_coords):
    """Convert one coordinate [x1, y1, x2, y2] to [x_center, y_center, w, h].
    It is the reverse process of ``obj_box_coord_centroid_to_upleft_butright``.
    Parameters
    ------------
    coord : list of 4 int/float
        One or multi coordinate.
    Returns
    -------
    list of 4 numbers
        New bounding box.
    """
    coord_or_coords = np.asarray(coord_or_coords)
    wh = coord_or_coords[..., 2:] - coord_or_coords[..., :2]
    xy = coord_or_coords[..., :2] + wh / 2
    xywh = np.concatenate([xy, wh], -1)
    return xywh


def xywh_to_x1y1wh(coord_or_coords):
    """
    Convert one coordinate [x_center, y_center, w, h] to [x, y, w, h].
    It is the reverse process of ``obj_box_coord_upleft_to_centroid``.
    Parameters
    ------------
    coord : list of 4 int/float
        One coordinate.
    Returns
    -------
    list of 4 numbers
        New bounding box.
    """
    coord_or_coords = np.asarray(coord_or_coords)
    coord_or_coords[..., :2] = coord_or_coords[..., :2] - coord_or_coords[..., 2:] / 2
    return coord_or_coords


def x1y1wh_to_xywh(coord_or_coords):
    """
    Convert one coordinate [x, y, w, h] to [x_center, y_center, w, h].
    It is the reverse process of ``obj_box_coord_centroid_to_upleft``.
    Parameters
    ------------
    coord : list of 4 int/float
        One coordinate.
    Returns
    -------
    list of 4 numbers
        New bounding box.
    """
    coord_or_coords = np.asarray(coord_or_coords)
    coord_or_coords[..., :2] = coord_or_coords[..., :2] + coord_or_coords[..., 2:] / 2
    return coord_or_coords


if __name__ == '__main__':
    shape = [100, 200]
    point_xywh = [60, 80, 30, 40]
    point_xywh_batch = [point_xywh, point_xywh]

    coord_yxhw = xywh2yxhw(point_xywh)
    coord_yxhw_batch = xywh2yxhw(point_xywh_batch)

    coord_scale = coord_pixelunit_to_scale(point_xywh, shape)
    coord_scale_batch = coord_pixelunit_to_scale(point_xywh_batch, shape)

    coord_pixelunit = coord_scale_to_pixelunit(coord_scale, shape)
    coord_pixelunit_batch = coord_scale_to_pixelunit(coord_scale_batch, shape)

    point_x1y1x2y2 = xywh_to_x1y1x2y2(point_xywh)
    point_x1y1x2y2_batch = xywh_to_x1y1x2y2(point_xywh_batch)

    point_x1y1wh = xywh_to_x1y1wh(point_xywh)
    point_x1y1wh_batch = xywh_to_x1y1wh(point_xywh_batch)

    point_xywh = x1y1x2y2_to_xywh(point_x1y1x2y2)
    point_xywh_batch = x1y1x2y2_to_xywh(point_x1y1x2y2_batch)

    point_xywh = x1y1wh_to_xywh(point_x1y1wh)
    point_xywh_batch = x1y1wh_to_xywh(point_x1y1wh_batch)

    print()
