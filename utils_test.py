import tensorflow as tf
import matplotlib.pyplot as plt


def test_model(model, images):
    size = (images.shape[1] // 4, images.shape[1] // 4)
    images_lr = tf.image.resize(images, size).numpy()
    images_hr = model.predict(images_lr)
    images = (images + 1) * .5
    images_lr = (images_lr + 1) * .5
    images_hr = (images_hr + 1) * .5
    for img, img_lr, img_hr in zip(images, images_lr, images_hr):
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis('off')
        fig.add_subplot(1, 3, 2)
        plt.imshow(img_lr)
        plt.axis('off')
        fig.add_subplot(1, 3, 3)
        plt.imshow(img_hr)
        plt.axis('off')
    plt.show()


def sr_plot(model, images):
    images_hr = model.predict(images)
    images_hr = tf.image.adjust_contrast(images_hr, 1.1).numpy()
    for img, img_hr in zip(images, images_hr):
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis('off')
        fig.add_subplot(1, 3, 2)
        plt.imshow(img_hr)
        plt.axis('off')
    plt.show()
