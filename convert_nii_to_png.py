from PIL import Image
import numpy as np
import os
import nibabel as nib
import cv2

path_to_file = 'Data\MICCAI_BraTS2020_ValidationData'
path_to_mask_output = 'Data\BRATS_20_Training_masks_png'
path_to_output = 'Data\BRATS_20_Validation_data_png'
base_name = 'BraTS20_Validation_0'

for i in range(10,100):

    ##include if converting segmentation mask
    '''
    mask = nib.load(os.path.join(path_to_file, base_name + str(i), base_name + str(i) + "_seg.nii.gz"))
    MASK = mask.get_fdata()
    #for BRATS mask convert label 4 to 3 as no label 3
    MASK[MASK == 4] = 3
    '''

    flair = nib.load(os.path.join(path_to_file, base_name + str(i), base_name + str(i) + "_flair.nii.gz"))
    t2 = nib.load(os.path.join(path_to_file, base_name + str(i), base_name + str(i) + "_t2.nii.gz"))
    t1 = nib.load(os.path.join(path_to_file, base_name + str(i), base_name + str(i) + "_t1.nii.gz"))
    t1c = nib.load(os.path.join(path_to_file, base_name + str(i), base_name + str(i) + "_t1ce.nii.gz"))
    FLAIR = flair.get_fdata()
    T2 = t2.get_fdata()
    T1 = t1.get_fdata()
    T1c = t1c.get_fdata()

    #normailse image I - median / interquartile range and then fit image so max is 1.5 x +/- IQR
    def norm_img(img):
        iMean = np.mean(img[img > 0])
        iSD = (np.std(img[img > 0]) * 3) / 128

        img[img > 0] = ((img[img > 0] - iMean) / iSD) + 127
        img[img < 0] = 0
        img[img > 255] = 255
        return img

    nFLAIR = norm_img(FLAIR)
    nT2 = norm_img(T2)
    nT1 = norm_img(T1)
    nT1c = norm_img(T1c)


    print(str(i))
    print(FLAIR.shape)
    H = 240
    W = 240
    slice = np.zeros([H,W,4])
    slice_m = np.zeros([H,W,3])

    for j in range(0,FLAIR.shape[2]):

        slice[:,:,0] = cv2.resize(nFLAIR[:,:,j].astype(np.uint8),(H,W))
        slice[:,:,1] = cv2.resize(nT1c[:,:,j].astype(np.uint8),(H,W))
        slice[:,:,2] = cv2.resize(nT1[:,:,j].astype(np.uint8),(H,W))
        slice[:,:,3] = cv2.resize(nT2[:,:,j].astype(np.uint8),(H,W))

        s_png = Image.fromarray(slice.astype(np.uint8))
        s_png.convert("RGBA")
        s_png.save(os.path.join(path_to_output, base_name + str(i) + "_slice_" + str(j) + ".png"))

        ##include if converting segmentation mask
        '''
        slice_m[:,:,0] = cv2.resize(MASK[:,:,j].astype(np.uint8),(H,W))
        slice_m[:, :, 1] = cv2.resize(MASK[:, :, j].astype(np.uint8), (H, W))
        slice_m[:, :, 2] = cv2.resize(MASK[:, :, j].astype(np.uint8), (H, W))
        sm_png = Image.fromarray(slice_m.astype(np.uint8))
        sm_png.convert("RGB")
        sm_png.save(os.path.join(path_to_mask_output, base_name + str(i) + "_slice_" + str(j) + ".png"))
        '''
