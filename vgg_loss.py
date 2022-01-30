import tensorflow as tf
from tensorflow.keras import layers as kl


def build_vgg(input_shape):
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    x_in = kl.Input(input_shape)
    x = vgg.layers[0](x_in)
    x = vgg.layers[1](x)
    x = vgg.layers[2](x)
    x = vgg.layers[3](x)
    x = vgg.layers[4](x)
    x = vgg.layers[5](x)
    x = vgg.layers[6](x)
    x = vgg.layers[7](x)
    x = vgg.layers[8](x)
    x = vgg.layers[9](x)
    # x = vgg.layers[10](x)
    # x = vgg.layers[11](x)
    # x = vgg.layers[12](x)
    # x = vgg.layers[13](x)
    # x = vgg.layers[14](x)
    # x = vgg.layers[15](x)
    # x = vgg.layers[16](x)
    # x = vgg.layers[17](x)
    # x = vgg.layers[18](x)
    # x = vgg.layers[19](x)
    # x = vgg.layers[20](x)
    # x = vgg.layers[21](x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x)
    model.trainable = False
    for layer in model.layers:
        layer.trainable = False
    return model


class VGGLoss():

    def __init__(self, input_shape):
        self.vgg = build_vgg(input_shape)
        self.mse = tf.keras.losses.MeanSquaredError() 

    def __call__(self, y_true, y_pred):
        loss = self.mse(self.vgg(y_true), self.vgg(y_pred))
        return loss
