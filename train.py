from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from Dr_Unet104_model import DR_Unet104
import numpy as np
from tensorflow.keras import backend as K

print('TensorFlow', tf.__version__)

# Set GPU memory limitation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 10GB of memory on the GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
# End of the GPU memory limitation code

#Set batch size, image size and number of segmentation classes (including background)
batch_size = 10
H, W = 240, 240
num_classes = 4

#set up list if traning PNG files from traning directory
image_list = sorted(glob(
    'Data/BRATS_20_Training_full_png/*'))
mask_list = sorted(glob(
    'Data/BRATS_20_Training_full_png_masks/*'))

##include below if using validation data during training such as sub set of BRATS training images
'''
val_image_list = sorted(glob(
    'Data/BRATS_20_Val_png/*'))
val_mask_list = sorted(glob(
    'Data/BRATS_20_Val_png_masks/*'))
'''

print('Found', len(image_list), 'training images')
'''
print('Found', len(val_image_list), 'validation images')
'''


def get_image(image_path, img_height=240, img_width=240, mask=False, flip=0, flip2=0):
    img = tf.io.read_file(image_path)

    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=4), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width]) / 255
        img_shape = tf.shape(img)
        print(img_shape)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = tf.case([
            (tf.greater(flip2, 0), lambda: tf.image.flip_up_down(img))
        ], default=lambda: img)
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
            img_height, img_width]), dtype=tf.uint8)
        img = tf.case([
            (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        ], default=lambda: img)
        img = tf.case([
            (tf.greater(flip2, 0), lambda: tf.image.flip_up_down(img))
        ], default=lambda: img)
        img = K.squeeze(img, axis=-1)
        img = K.one_hot(tf.cast(img, tf.int32), num_classes)
    return img


def load_data(image_path, mask_path):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.float32)[0]
    flip2 = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.float32)[0]
    image, mask = get_image(image_path, flip=flip, flip2=flip2), get_image(
        mask_path, mask=True, flip=flip, flip2=flip2)
    return image, mask


train_dataset = tf.data.Dataset.from_tensor_slices((image_list,
                                                    mask_list))
train_dataset = train_dataset.shuffle(buffer_size=128)
train_dataset = train_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(train_dataset)

##only include if using validation images during traning
'''
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,
                                                  val_mask_list))
val_dataset = val_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
'''

##set loss function
loss = tf.losses.CategoricalCrossentropy(from_logits=True)

def dice_metric(y_true, y_pred, smooth=1):
    pred_tensor = tf.argmax(y_pred, axis=3)
    y_pred = K.one_hot(tf.cast(pred_tensor, tf.int32), 4)
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    union = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DR_Unet104(H,W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.99
            layer.epsilon = 1e-5
    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=[dice_metric])

tb = TensorBoard(log_dir='logs_dr_unet104', write_graph=True, update_freq='epoch')
mc = ModelCheckpoint(mode='min', filepath='weights_brats_training_dr_unet104.h5',
                     monitor='loss',
                     # save_best_only='True',
                     save_weights_only='True', verbose=1)
callbacks = [mc, tb]

##to load weights for further training
'''
checkpoint_path=r'weights_brats_training_dr_unet104.h5'
model.load_weights(checkpoint_path, by_name=True)
'''

Hx = model.fit(train_dataset,
               steps_per_epoch=len(image_list) // batch_size,
               epochs=50,
               ##include below if using validation data
               #validation_data=val_dataset,
               #validation_steps=len(val_image_list) // batch_size,
               callbacks=callbacks)

##save training loss and dice metric
train_loss = Hx.history['loss']
train_losses = np.array(train_loss)
np.savetxt("train_brats_loss.txt", train_losses, delimiter=",")
train_metric = Hx.history['dice_metric']
train_losses = np.array(train_metric)
np.savetxt("train_brats_dice_metric.txt", train_metric, delimiter=",")

##to save validation training loss and metric
'''
validate_loss = Hx.history['val_loss']
val_losses = np.array(validate_loss)
np.savetxt("train_brats_loss_val.txt", val_losses, delimiter=",")
validate_metric = Hx.history['dice_metric']
val_metric = np.array(validate_metric)
np.savetxt("train_brats_dice_metric_val.txt", val_losses, delimiter=",")
'''
