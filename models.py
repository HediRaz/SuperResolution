import tensorflow as tf
import tensorflow.keras.layers as kl


class res_block(tf.keras.Model):
    def __init__(self, nb_filters1=128, nb_filters2=128, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(nb_filters1, 3, 1, 'same')
        self.bn1 = kl.BatchNormalization()
        self.acti1 = kl.PReLU(shared_axes=[1, 2])
        self.conv2 = kl.Conv2D(nb_filters2, 3, 1, 'same')
        self.bn2 = kl.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.acti1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x += input_tensor
        return x


class generator(tf.keras.Model):
    def __init__(self, B, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(64, 9, 1, padding="same")
        self.acti1 = kl.PReLU(shared_axes=[1, 2])

        self.res_blocks = tf.keras.Sequential()
        for _ in range(B):
            self.res_blocks.add(res_block(64, 64))

        self.conv2 = kl.Conv2D(64, 3, 1, padding="same")
        self.bn2 = kl.BatchNormalization()

        self.conv3 = kl.Conv2D(256, 3, 1, padding="same")
        self.up1 = lambda x: tf.nn.depth_to_space(x, 2)
        self.acti3 = kl.PReLU(shared_axes=[1, 2])

        self.conv4 = kl.Conv2D(256, 3, 1, padding="same")
        self.up2 = lambda x: tf.nn.depth_to_space(x, 2)
        self.acti4 = kl.PReLU(shared_axes=[1, 2])

        self.conv5 = kl.Conv2D(3, 9, 1, padding="same")

    def call(self, x):
        x = self.conv1(x)
        x = self.acti1(x)

        x1 = self.res_blocks(x)
        x1 = self.conv2(x1)
        x = self.bn2(x1) + x

        x = self.conv3(x)
        x = self.up1(x)
        x = self.acti3(x)

        x = self.conv4(x)
        x = self.up2(x)
        x = self.acti4(x)

        x = self.conv5(x)
        return tf.tanh(x)


class conv_block(tf.keras.Model):
    def __init__(self, nb_filters, stride, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(nb_filters, 3, stride, 'same')
        self.bn1 = kl.BatchNormalization()
        self.acti1 = kl.LeakyReLU()

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = self.acti1(x)
        return x


class discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = kl.Conv2D(64, 9, 1, padding="same")
        self.acti1 = kl.LeakyReLU()

        self.block1 = conv_block(64, 2)  # 64
        self.block2 = conv_block(128, 1)
        self.block3 = conv_block(128, 2)  # 32
        self.block4 = conv_block(256, 1)
        self.block5 = conv_block(256, 2)  # 16
        self.block6 = conv_block(512, 1)
        self.block7 = conv_block(512, 2)  # 8

        self.flatten = kl.Flatten()

        self.dense1 = kl.Dense(512)
        self.a1 = kl.LeakyReLU()

        self.dense2 = kl.Dense(1)

    def call(self, x):
        x = self.conv1(x)
        x = self.acti1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.a1(x)
        x = self.dense2(x)
        return x  # tf.sigmoid(x)
