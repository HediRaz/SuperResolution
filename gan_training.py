import tensorflow as tf
import numpy as np
from models import generator, discriminator
from utils_images import create_dataset
from utils_test import test_model


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
optimizer = tf.keras.optimizers.Adam()

train_folder_path = "Dataset/imagenet/train_processed_128"
val_folder_path = "Dataset/imagenet/val_processed_128"
train_dataset = create_dataset(train_folder_path, batch_size=BATCH_SIZE, input_size=(128, 128, 3), shuffle=True)
val_dataset = create_dataset(val_folder_path, batch_size=1, input_size=(128, 128, 3), shuffle=True)

test_images = np.array([a[1][0].numpy() for a in val_dataset])

gen = generator(B=16)
# gen.build((None, None, None, 3))
# gen.summary()
dis = discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()
# mse = tf.keras.losses.MeanAbsoluteError()
gen_ce_tracker = tf.keras.metrics.Mean(name="gen_ce")
dis_ce_tracker = tf.keras.metrics.Mean(name="dis_ce")
mse_tracker = tf.keras.metrics.Mean(name="mse")
acc_fake_tracker = tf.keras.metrics.Mean(name="acc_fake")
acc_real_tracker = tf.keras.metrics.Mean(name="acc_real")


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    dis_ce_tracker.update_state(total_loss)
    acc = tf.reduce_sum(tf.where(real_output > 0.5, 1, 0)) / BATCH_SIZE
    acc_real_tracker.update_state(acc)
    acc = tf.reduce_sum(tf.where(fake_output < 0.5, 1, 0)) / BATCH_SIZE
    acc_fake_tracker.update_state(acc)
    return total_loss


def generator_loss(fake_output, generated_images, images):
    ce = cross_entropy(tf.ones_like(fake_output), fake_output)
    gen_ce_tracker.update_state(ce)
    mse_value = mse(images, generated_images)
    mse_tracker.update_state(mse_value)
    return (1e-3 * ce) + mse_value


@tf.function
def train_step(gen, dis, images_lr, images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(images_lr, training=True)

        real_output = dis(images, training=True)
        fake_output = dis(generated_images, training=True)

        gen_loss = generator_loss(fake_output, generated_images, images)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, dis.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, dis.trainable_variables))


def train(gen, dis, dataset, epochs):
    num_batch = dataset.cardinality()
    for epoch in range(epochs):
        for k, (images_lr, images) in enumerate(dataset):
            if k == 1000:
                break
            train_step(gen, dis, images_lr, images)
            tf.print(f"[{k:^5}/{num_batch}]  mse: {mse_tracker.result():4f}  gen_ce: {gen_ce_tracker.result():4f}  dis_ce: {dis_ce_tracker.result():4f}  acc_real: {acc_real_tracker.result():4f}  acc_fake: {acc_fake_tracker.result():4f}")


train(gen, dis, train_dataset, 1)

print("Test")
test_model(gen, test_images)
