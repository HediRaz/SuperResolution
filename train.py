import os
import tensorflow as tf
import numpy as np
from models import generator
from utils_images import CustomDataGen, create_dataset
from utils_test import test_model
from vgg_loss import VGGLoss


EPOCHS = 1
BATCH_SIZE = 16
train_folder_path = "Dataset/imagenet/train_processed_128"
val_folder_path = "Dataset/imagenet/val_processed_128"

B = 16
optimizer = tf.keras.optimizers.Adam()

loss = tf.keras.losses.MeanSquaredError()
# loss = VGGLoss((128, 128, 3))


train_dataset = create_dataset(train_folder_path, batch_size=BATCH_SIZE, input_size=(128, 128, 3), shuffle=True)
val_dataset = create_dataset(val_folder_path, batch_size=1, input_size=(128, 128, 3), shuffle=True)

model = generator(B)
model.build((None, None, None, 3))
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[lambda x, y: tf.image.psnr(x, y, 1)]
)

model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
model.save_weights("SavedModels/generator")

print("Test")
test_images = np.array([a[1][0].numpy() for a in val_dataset])
test_model(model, test_images)
