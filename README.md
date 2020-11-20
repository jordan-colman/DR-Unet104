# DR-Unet104
Deep residual Unet with 104 convolutional layers 

DR-Unet104 is based on the Unet and uses deep residual bottleneck blocks.
The network takes 2D png input and outputs a pixel wise semantic segmentation. Code is aditionally included to normalise and convert nifti (.nii) format MRI images to seperate 2D png slices (convert_nii_to_png.py) and then convert the network output segmentations to 3D .nii files (convert_png_mask_to_nii.py). 

The U-net based encoder-decoder design can be seen below. Multiple stacks of bottleneck blocks, taken from ResNet, are used in the networks encoder. The blocks are stacked, 2, 3, 4, 5, or 14 times to give 104 convolutional layers in the whole network in total.

![alt text](https://user-images.githubusercontent.com/67955222/99743883-8486fa00-2ace-11eb-990c-316873ff32cd.png)

The bottleneck residual block and typical residual blocks are further outlined below. The bottlenck block reduces the number of feature channels with a 1x1 convolution prior to the 3x3 spatial convolution and then expands the number of feature channels to match the input size with a final 1x1 convolution. This increases the number of feature channels for the same computational cost. The network blocks take pre-activated input to aid backpropogation though the residual connections as used in ResNetV2.

![alt text](https://user-images.githubusercontent.com/67955222/99744454-a7fe7480-2acf-11eb-9eef-83575bb5c8e8.png)


This network was evaluated as part of the Multimodal Brain Tumor Segmentation (BraTS) 2020 challenge (http://braintumorsegmentation.org/) an example of the traning data can be seen below.

![alt text](https://user-images.githubusercontent.com/67955222/99747410-5dcac280-2ad2-11eb-8431-6d79bf6b9d35.png)

Find trained model weights trained on the BraTS 2020 training data here: https://drive.google.com/file/d/1v8b5dPEa9nTVVkzjdjX6exZo4anlOhTH/view?usp=sharing 

You are able to use this network for your own research if you cite our paper outlining the arcitecture:
'DR-Unet104 for Multimodal MRI brain tumor segmentation', Jordan Colman, Lei Zhang, Wenting Duan and Xujiong Ye (2020), arxiv 2011.02840

 pre-print avalible at: https://arxiv.org/abs/2011.02840
