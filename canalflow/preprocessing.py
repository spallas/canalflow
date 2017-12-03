
import tensorflow as tf
import os


def input_from_dataset():

    src_dir = "../data/train"
    file_names = [src_dir + "/" + name.strip() for name in os.listdir(src_dir)]
    # print(file_names[:6])
    dataset = tf.data.TFRecordDataset(file_names)

    num_epochs = 10

    def parser(record):
        keys_to_features = {
            "image/class/label": tf.FixedLenFeature([], tf.int64, default_value=-1),
            "image/class/text": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/encoded": tf.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional pre-processing on the parsed data.
        image = tf.cast(tf.image.decode_jpeg(parsed["image/encoded"]), tf.float32)
        image = tf.reshape(image, [800, 240, 3])
        label = tf.cast(parsed["image/class/label"], tf.int32)

        return image, label

    dataset = dataset.map(parser)\
                     .shuffle(buffer_size=10000)\
                     .batch(32)\
                     .repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()

    return features, labels


def load_test_set():
    pass
