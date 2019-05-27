# virtual dataset class


class Dataset:
    def __init__(self):
        pass

    def _not_imp(self):
        raise NotImplementedError('This is abstact function')


# get the dataset information
    def get_label_num(self):
        """
        how many label in this dataset
        :return:
        """
        self._not_imp()

    def get_class_num(self):
        """
        how many class in this dataset
        :return:
        """
        self._not_imp()

    def get_class_name(self):
        """
        get classes name
        :return:
        """
        self._not_imp()

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
        [ image_path, class_num, instance_num ]
        :param label_id:
        :return:
        """
        self._not_imp()

    def get_label_image(self, label_id):
        """
        get origin image
        :param label_id:
        :return:
        """
        self._not_imp()

    def get_label_instance_bbox(self, label_id):
        """
        for object detection
        :param label_id:
        :return:
        """
        self._not_imp()

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

    def get_label_instance_keypoints(self, label_id):
        """
        for person keypoints
        :param label_id:
        :return:
        """
        self._not_imp()
