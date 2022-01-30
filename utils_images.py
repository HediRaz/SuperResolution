import tensorflow as tf
import random
import os


def make_path_dataset(folder):
    imgs_paths = [os.path.join(folder, img_filename) for img_filename in os.listdir(folder)]
    imgs_paths = tf.constant(imgs_paths)
    path_ds = tf.data.Dataset.from_tensor_slices(imgs_paths)
    return path_ds


def load_jpeg_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    return img


def normalize_fn(img):
    return (img/127.5) - 1


def alteration_fn(img, size):
    return tf.image.resize(img, (size[0]//4, size[1]//4))


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, data_paths: list, batch_size: int, input_size=(128, 128, 3), shuffle=True):

        self.data_paths = data_paths.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.data_paths)

        self.n = len(self.data_paths)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_paths)

    def __getitem__(self, idx):
        y = tf.stack(
            [load_jpeg_img(self.data_paths[i]) for i in range(idx*self.batch_size, (idx+1)*self.batch_size)],
            0
        )
        y = normalize_fn(y)
        X = alteration_fn(y, self.input_size)
        return X, y

    def __call__(self):
        return self

    def __len__(self):
        return self.n // self.batch_size


def create_dataset(data_folder, batch_size: int, input_size=(128, 128, 3), shuffle=True):
    data_paths = [os.path.join(data_folder, e) for e in os.listdir(data_folder)]
    generator = CustomDataGen(data_paths, batch_size, input_size, shuffle)
    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, input_size[0]//4, input_size[1]//4, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, input_size[0], input_size[1], 3), dtype=tf.float32)))
    dataset = tf.data.experimental.assert_cardinality(len(generator))(dataset)
    return dataset
