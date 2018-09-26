from __future__ import print_function, division, absolute_import
import numpy as np
import cv2


class ImageSplitter(object):
    def __init__(self, image, scale, color_switch, crop_image_size, IMG_MEAN):
        height, width = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        scaled_height, scaled_width = int(height * scale), int(width * scale)
        image = cv2.resize(image, dsize=(scaled_width, scaled_height))  # dsize should be (width, height)
        image -= IMG_MEAN
        if color_switch:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # if switch, image: bgr

        # print image.shape  # (1024, 2048, 3) (512, 1024, 3)
        self.dif_height = scaled_height - crop_image_size
        self.dif_width = scaled_width - crop_image_size
        if self.dif_height < 0 or self.dif_width < 0:
            image = numpy_pad_image(image, self.dif_height, self.dif_width)
            scaled_height, scaled_width = image.shape[0], image.shape[1]
        # print image.shape  # (1024, 2048, 3) (864, 1024, 3)
        self.image = image
        self.crop_image_size = crop_image_size

        self.heights = decide_intersection(scaled_height, crop_image_size)
        self.widths = decide_intersection(scaled_width, crop_image_size)

        split_crops = []
        for height in self.heights:
            for width in self.widths:
                image_crop = self.image[height:height + crop_image_size, width:width + crop_image_size]
                split_crops.append(image_crop[np.newaxis, :])

        self.split_crops = np.concatenate(split_crops, axis=0)  # (n, crop_image_size, crop_image_size, 3)

    def get_split_crops(self):
        return self.split_crops

    def reassemble_crops(self, proba_crops):
        # in general, crops are probabilities of self.split_crops.
        assert proba_crops.shape[0:3] == self.split_crops.shape[0:3], \
            '%s vs %s' % (proba_crops.shape[0:3], self.split_crops.shape[0:3])
        # (n, crop_image_size, crop_image_size, num_classes) vs (n, crop_image_size, crop_image_size, 3)

        # reassemble
        reassemble = np.zeros((self.image.shape[0], self.image.shape[1], proba_crops.shape[-1]), np.float32)
        index = 0
        for height in self.heights:
            for width in self.widths:
                reassemble[height:height+self.crop_image_size, width:width+self.crop_image_size] += proba_crops[index]
                index += 1
        # print reassemble.shape

        # crop to original image
        if self.dif_height < 0 or self.dif_width < 0:
            reassemble = numpy_crop_image(reassemble, self.dif_height, self.dif_width)

        return reassemble


def numpy_crop_image(image, dif_height, dif_width):
    # (height, width, channel)
    assert len(image.shape) == 3
    if dif_height < 0:
        if dif_height % 2 == 0:
            pad_before_h = - dif_height // 2
            pad_after_h = dif_height // 2
        else:
            pad_before_h = - dif_height // 2
            pad_after_h = dif_height // 2
        image = image[pad_before_h:pad_after_h]

    if dif_width < 0:
        if dif_width % 2 == 0:
            pad_before_w = - dif_width // 2
            pad_after_w = dif_width // 2
        else:
            pad_before_w = - dif_width // 2
            pad_after_w = dif_width // 2
        image = image[:, pad_before_w:pad_after_w]

    return image


def decide_intersection(total_length, crop_length):
    stride = crop_length * 2 // 3
    times = (total_length - crop_length) // stride + 1
    cropped_starting = []
    for i in range(times):
        cropped_starting.append(stride*i)
    if total_length - cropped_starting[-1] > crop_length:
        cropped_starting.append(total_length - crop_length)
    return cropped_starting


def compute_confusion_matrix(label, prediction, confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    label = np.reshape(label, (-1))
    prediction = np.reshape(prediction, (-1))
    indice = np.where(label < num_classes)
    label = label[indice]
    prediction = prediction[indice]

    np.add.at(confusion_matrix, (label, prediction), 1)
    return confusion_matrix


def compute_iou(confusion_matrix):
    cm = confusion_matrix.astype(np.float)
    sum_over_row = np.sum(cm, 0)
    sum_over_col = np.sum(cm, 1)
    cm_diag = np.diag(cm)
    denominator = sum_over_row + sum_over_col - cm_diag

    iou = np.divide(cm_diag, denominator)
    for i in range(confusion_matrix.shape[0]):
        print('%.1f' % (iou[i]*100), '&', end='')
    print('%.1f' % (np.mean(iou)*100), '\\\\')
    return np.mean(iou)


def compute_iou_each_class(confusion_matrix):
    cm = confusion_matrix.astype(np.float)
    sum_over_row = np.sum(cm, 0)
    sum_over_col = np.sum(cm, 1)
    cm_diag = np.diag(cm)
    denominator = sum_over_row + sum_over_col - cm_diag
    return np.divide(cm_diag, denominator)


def numpy_pad_image(image, total_padding_h, total_padding_w, image_padding_value=0):
    # (height, width, channel)
    assert len(image.shape) == 3
    pad_before_w = pad_after_w = 0
    pad_before_h = pad_after_h = 0
    if total_padding_h < 0:
        if total_padding_h % 2 == 0:
            pad_before_h = pad_after_h = - total_padding_h // 2
        else:
            pad_before_h = - total_padding_h // 2
            pad_after_h = - total_padding_h // 2 + 1
    if total_padding_w < 0:
        if total_padding_w % 2 == 0:
            pad_before_w = pad_after_w = - total_padding_w // 2
        else:
            pad_before_w = - total_padding_w // 2
            pad_after_w = - total_padding_w // 2 + 1
    image_crop = np.pad(image,
                        ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
                        mode='constant', constant_values=image_padding_value)
    return image_crop


if __name__ == '__main__':
    print(decide_intersection(int(2048 * 1.75), 864))
    assert decide_intersection(int(2048 * 1.75), 864) == [0, 576, 1152, 1728, 2304, 2720]
    assert decide_intersection(int(1024 * 1.75), 864) == [0, 576, 928]
    assert decide_intersection(int(2048 * 0.5), 864) == [0, 160]
    # print decide_intersection(int(1024 * 0.5), 864)
    assert decide_intersection(864, 864) == [0]
    assert decide_intersection(2048, 720) == [0, 480, 960, 1328]

    assert decide_intersection(875, 480) == [0, 320, 395]
    assert decide_intersection(800, 480) == [0, 320]
    assert decide_intersection(int(799.75), 480) == [0, 319]
    img = cv2.imread('/home/jacques/workspace/database/SDB_all/JPEGImages/2009_001314.jpg')
    splitter = ImageSplitter(img, 1.75, 0, 480, [0, 0, 0])

    # r_img = splitter.reassemble_crops(splitter.get_split_crops())

