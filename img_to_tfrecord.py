#import libraries
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_img_example(img):
    """
    create a single tf.train.Example obj instance given a single input sample
    in the form of sequence & labels pair.
    """
    height, width = img.shape[:2]
    example = tf.train.Example(
        features=tf.train.Features(feature={
            "height": _int64_feature([height]),
            "width": _int64_feature([width]),
            "image": _bytes_feature([tf.compat.as_bytes(img.tostring())])
        })
    )
    return example


def make_tfrecord(img_dir, outf_nm='my_dataset'):
    """
    data is in the format of tuple (image, labels), where each image and labels
    are list objects of image arrays and one-hot labels, respectively
    """
    img_file_list = glob(img_dir)
    outf_nm += '.tfrecord'
    tfrecord_wrt = tf.python_io.TFRecordWriter(outf_nm)
    for img_file in tqdm(img_file_list):
        # read image into np.array
        img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        # compute shrink ratio
        shrink_ratio = TGT_SHAPE/max(img.shape)
        # shrink image to reduce size in resultant tf record file
        img = cv2.resize(img, (0, 0), fx=shrink_ratio, fy=shrink_ratio)
        # pass to function to serialize to tf example
        exmp = make_img_example(img)
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)


TGT_SHAPE = 128


if __name__ == "__main__":
    # Example usage
    make_tfrecord(img_dir='dataset/img_align_celeba/*.jpg',
                  outf_nm='dataset/celeba')
