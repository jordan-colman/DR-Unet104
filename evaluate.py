import numpy as np
from tensorflow.python.keras.preprocessing.image import img_to_array
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
from Dr_Unet104_model_github import DR_Unet104

# Set GPU memory limitation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 10GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# End of the GPU memory limitation code

h, w = 240, 240

model = DR_Unet104(h,w,4)
model.load_weights('weights_brats_dr_unet_test_model_DO_05.h5')

def get_image(image_path, img_height=240, img_width=240):
    img = tf.io.read_file(image_path)

    img = tf.cast(tf.image.decode_png(img, channels=4), dtype=tf.float32)
    img = tf.image.resize(images=img, size=[img_height, img_width]) / 255
    return img


def load_data(image_path, H=256, W=256):
    image = get_image(image_path)
    return image


def pipeline(image):
    dims = image.shape
    print(dims)
    image = cv2.resize(image, (w, h))
    x = image.copy()
    z = model.predict(np.expand_dims(x, axis=0))

    print(z.shape)
    z = np.squeeze(z)

    y = np.argmax(z, axis=-1)

    img_color = image.copy()
    img_prob = np.zeros([h,w,3])
    print(img_color.shape)

    img_color[:, :, 0] = y
    img_color[:, :, 1] = 0
    img_color[:, :, 2] = 0

    return img_color * 50

image_dir = 'Data/BRATS_20_Val_full_png'
image_list = os.listdir(image_dir)
image_list.sort()
print(f'{len(image_list)} frames found')

for i in tqdm(range(len(image_list))):
    test = load_data(f'{image_dir}/{image_list[i]}')
    test = img_to_array(test)
    segmap = pipeline(test)
    fname = f'{image_list[i]}'
    cv2.imwrite(f'Data/BRATS_20_Validation_mask_results_png/{fname}', cv2.cvtColor(segmap, cv2.COLOR_RGB2BGR))
