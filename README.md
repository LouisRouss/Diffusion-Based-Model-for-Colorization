# WORK IN PROGRESS ( Not any trained model available yet / Training in progress )


# About this project 
This is a first implementation of a Colorization Diffusion Based Method

# How to train the model
Modify the conf.yml file, set the 'mode' option to 1. Then run the main.py file specifying the path to the config file  (absolute or relative)
Example : python main.py --config conf.yml

# Future of this repo
I have impossibility to train and to test the model implemented due to my lack of computational power.
There might be some mistakes in the code, any insight and remark is welcomed

For the validation loop in the training loop, necessity to use/find a more suitable/ an additional metric

# Reference
Palette Image_to_Image Diffusion Models https://arxiv.org/pdf/2111.05826v1.pdf

Diffusion Models Beat GANs on Image Synthesis https://arxiv.org/pdf/2105.05233.pdf 

The Unet Network script directly comes from the repo of this last : https://github.com/openai/guided-diffusion (with small modifications according to the Palette paper)

A colorization Dataset : https://www.kaggle.com/shravankumar9892/image-colorization ( Palette paper's researchers uses ImageNet )
