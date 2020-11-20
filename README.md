# DR-Unet104
Deep residual Unet with 104 convolutional layers 

DR-Unet104 is based on the Unet and uses deep residual bottleneck blocks.
The U-net based encoder decoder design can be seen below. Multiple stacks of bottleneck blocks, as used in the ResNet50 or ResNet101, are used in the networks encoder. The blocks are stacked, 2, 3, 4, 5, or 14 times in the encoder to give 104 convolutional layers in the whole network in total.

![alt text](https://user-images.githubusercontent.com/67955222/99743883-8486fa00-2ace-11eb-990c-316873ff32cd.png)

The bottleneck residual block and typical residual blocks are further outlined below. The bottlenck block reduces the number of feature channels prior to the spatial convolution and there expnds the number of featuch channels. This increases the number in feature channels for the same computational cost. The network blocks take pre-activated input to aid backpropogation though the residual connections as used in ResNetV2.

![alt text](https://user-images.githubusercontent.com/67955222/99744454-a7fe7480-2acf-11eb-9eef-83575bb5c8e8.png)


This network was evaluated as part of the Multimodal Brain Tumor Segmentation (BraTS) 2020 challenge
http://braintumorsegmentation.org/

Find trained model weights trained on the BraTS 2020 training data here: https://drive.google.com/file/d/1v8b5dPEa9nTVVkzjdjX6exZo4anlOhTH/view?usp=sharing 

You are able to use this network for your own research if you cite our paper outline the arcitecture:
'DR-Unet104 for Multimodal MRI brain tumor segmentation', Jordan Colman, Lei Zhang, Wenting Duan and Xujiong Ye (2020), arxiv 2011.02840

 pre-print avalible at: https://arxiv.org/abs/2011.02840
