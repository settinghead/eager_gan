import tensorflow as tf
import math


def save_model(root, checkpoint_prefix):
    root.save(file_prefix=checkpoint_prefix)


def tf_record_parser(record):
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64)
    }

    features = tf.parse_single_example(record, keys_to_features)

    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.decode_raw(features['image'], tf.uint8)

    # reshape input and annotation images
    image = tf.reshape(image, (height, width, 3), name="image_reshape")
    image = tf.image.resize_image_with_crop_or_pad(image, 128, 128)

    image = tf.image.random_flip_left_right(image)
    # image = tf.contrib.image.rotate([image], tf.random_uniform(
    #     [1], maxval=math.pi / 10))[0]
    return tf.to_float(image)


def normalizer(image, dtype):
    # Not sure which one works better yet
    image = tf.cast(image, dtype=dtype) / 255.0 - 0.5
    # image = tf.cast(image, dtype=dtype) / 128.0 - 1.0
    # noise addition normalization
    image += tf.random_uniform(shape=tf.shape(image),
                               minval=0., maxval=1./128., dtype=dtype)

    return image
