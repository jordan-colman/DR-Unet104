import numpy as np
import nibabel as nib
import cv2
import os

path_to_flair = 'Data\MICCAI_BraTS2020_ValidationData'
path_to_file = 'Data\BRATS_20_Validation_mask_results_png'
path_to_out = 'Data\BRATS_20_Val_full_results_mask_nii'
base_name = 'BraTS20_Validation_0'

for i in range(10,100):
    orig_img = nib.load(os.path.join(path_to_flair, base_name + str(i), base_name + str(i) + '_flair.nii.gz'))
    orig_img_data = orig_img.get_fdata()
    zlen = orig_img_data.shape[2]
    w = orig_img_data.shape[0]
    h = orig_img_data.shape[1]
    array_3d = np.zeros([w,h,zlen])
    print(array_3d.shape)
    for j in range(0,(zlen)):
        name = os.path.join(base_name + str(i) + '_slice_')
        slice = j
        img_name = os.path.join(path_to_file, name + str(slice) + '.png')
        print(img_name)
        im_frame = cv2.imread(img_name)
        np_frame = np.array(im_frame)
        print(np_frame.shape)
        array_3d[:,:,j] = cv2.resize(np.copy(np_frame[:,:,2]),(h,w))
    array_3d = array_3d / 50
    array_3d[array_3d == 3] = 4

    img = nib.Nifti1Image(array_3d, orig_img.affine, header=orig_img.header)


    out_file = os.path.join(path_to_out, base_name + str(i) + '.nii.gz')
    nib.save(img, out_file)
