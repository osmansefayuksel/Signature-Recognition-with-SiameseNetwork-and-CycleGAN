# Signature Recognition with SiameseNetwork and CycleGAN

The signature is one of the most popular and widely accepted biometric signs used to verify human differences since ancient times. Therefore, signature verification is a critical task and many efforts have been made to eliminate the ambiguity in the manual authentication procedure, which has made signature verification an important research area in the field of machine learning. One of these can be modeled by a Siamese network consisting of convolutional neural networks that accept two different signature images.

<br>

## What is a Siamese Neural Network?

Siamese neural networks are the combination of two or more neural networks of the same structure. These networks share the same weights and parameters and are trained with the same dataset. These structures are used to solve problems such as similarity or matching.<br>
Siamese neural networks take two different inputs and use them to compare. These two inputs can be taken from the same dataset or different datasets.
For example, in an image dataset, it can be used to measure the similarity of two images.<br>
The output of Siamese neural networks can be a score that measures the similarity of two inputs. This score can be greater than zero and equal to one, and values between it indicate the similarity of the two entries. The value of the score indicates how similar two entries are.
<br> 
<br>

## What are Generative Adversarial Networks (GAN)?

In GANs, there are two different artificial neural networks competing with each other. These are called Generator and Discriminator networks. While the Generative network produces new data that resembles reality (images, sounds, models, etc.), the Discriminative network tries to distinguish between fake and real data. As these two neural networks compete with each other, the Discriminative network gradually begins to better distinguish real and fake images. The generative network produces more realistic fake images.
<br>
<br>

![Generative Adversarial Networks](https://github.com/osmansefayuksel/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/blob/main/images/fig1.png)

<br>
<br>

## Data Preprocessing for CycleGAN

Signatures in real-world documents often contain noise-induced artifacts such as stamps/seals, text, and printed lines. These noise occurrences can affect the signature verification process. A noise cleansing method based on CycleGAN will be applied on the detected signatures to create noise-free signatures. The CycleGAN model is trained using the Kaggle Signature Dataset. Noisy signatures are generated from the dataset using OpenCV.

<br>
<br>

## CycleGAN Result
<br>

![Input](https://github.com/osmansefayuksel/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/blob/main/results/cyclegan/001_16_real.png)

<br>

![Output](https://github.com/osmansefayuksel/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/blob/main/results/cyclegan/001_16_fake.png)

<br>
<br>

## Siamese Network
<br>

![SigNet](https://github.com/osmansefayuksel/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/blob/main/images/fig2.png)
<br>
<br>
The input layer, i.e. the 11x11 convolution layer with ReLU, is shown in blue, while all the 3x3 and 5x5 convolution layers are shown in cyan and green, respectively. All local response normalization layers are shown in magenta, all maximum pooling layers are shown in brick color, and dropout layers are shown in gray. The last orange block represents the high-level feature output from the generating CNNs, which is combined with the loss function in the equation.


## Final Result
<br>

![Recognition](https://github.com/osmansefayuksel/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/blob/main/images/fig3.png)
<br>


![Difference Score](https://github.com/osmansefayuksel/Signature-Recognition-with-SiameseNetwork-and-CycleGAN/blob/main/images/fig4.png)

<br>
<br>

## [CycleGAN](https://github.com/junyanz/CycleGAN)
## [Dataset](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)
## [Reference 1](https://www.kaggle.com/code/surveshchauhan/gl-cv-week-siamese-network-signature-verification)
## [Reference 2](https://github.com/amaljoseph/Signature-Verification_System_using_YOLOv5-and-CycleGAN)
## [SigNet Document](https://arxiv.org/abs/1707.02131v2)





