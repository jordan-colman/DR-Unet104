# DR-Unet104
Deep residual Unet with 104 convolutional layers 

DR-Unet104 is based on the Unet and uses deep residual bottleneck blocks
The U-net based encoder decoder design can be seen below. Multiple stacks of bottelneck blocks as used in the ResNet50 or ResNet101 are used in the networks encoder.

![alt text](https://user-images.githubusercontent.com/67955222/99743883-8486fa00-2ace-11eb-990c-316873ff32cd.png)

The bottelneck residual block and typical resnet block are further outlined below.



This network was evaluated as part of the Multimodal Brain Tumor Segmentation (BraTS) 2020 challenge
http://braintumorsegmentation.org/

Find trained model weights trained on the BraTS 2020 training data here: https://drive.google.com/file/d/1v8b5dPEa9nTVVkzjdjX6exZo4anlOhTH/view?usp=sharing 

You are able to use this network for your own research if you cite our research:
'DR-Unet104 for Multimodal MRI brain tumor segmentation', Jordan Colman, Lei Zhang, Wenting Duan and Xujiong Ye (2020), arxiv 2011.02840

 pre-print avalible at: https://arxiv.org/abs/2011.02840
