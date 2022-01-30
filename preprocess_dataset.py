import tensorflow as tf
import os
from tqdm import tqdm


def preprocess(folder: str, size: int, max_images=None):
    train_folder = os.path.normpath(folder)
    train_processed_folder = os.path.normpath(folder+"_processed_"+str(size))
    img_filenames = os.listdir(train_folder)
    max_images = len(img_filenames) if max_images is None else max_images
    for img_filename in tqdm(img_filenames[:max_images]):
        img_origin = tf.constant(os.path.join(train_folder, img_filename))
        img_final = tf.constant(os.path.join(train_processed_folder, img_filename))

        img = tf.io.read_file(img_origin)
        img = tf.image.decode_jpeg(img, channels=3)
        if img.shape[0] < size or img.shape[1] < size:
            continue
        img = tf.image.random_crop(img, [size, size, 3])

        img = tf.image.encode_jpeg(img)
        tf.io.write_file(img_final, img)


if __name__ == "__main__":
    preprocess("Dataset/imagenet/train", 128)
    preprocess("Dataset/imagenet/val", 128, 20)
