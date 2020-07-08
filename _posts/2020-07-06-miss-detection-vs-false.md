---
layout: post
title: Miss Detection vs. False Alarm: Adversarial Learning for Small Object Segmentation in Infrared Images
subtitle: Adversarial Learning
cover-img: /assets/img/path.jpg
tag: [infrared, cGAN, ISOS]
---



## Brief Review

"Miss Detection vs. False Alarm: Adversarial Learning for Small Object Segmentation in Infrared Images" is the paper on infrared small object segmentation (ISOS). Different from small object segmentation on visible image, infrared radiation will decay with distance, making the objects appear to be extremely dim. This makes this problem much difficult.

A key challenge of small object segmentation is to balane miss detection (MD) and false alarm(FA). Both MD and FA are depending on the same threshold.  The larger threshold cause the less MD but more FA. To balance MD and FA, author propose an cGAN network.



## Implementation Method

![IMG_D1611FD2005D-1](/Users/suyanyuan/Downloads/IMG_D1611FD2005D-1.jpeg)

### Model

Proposal model is based on cGAN. It has two generators G and a discriminator $D$. Each of the generators maps an input image $I$ to another image $S$ showing the segmentation result, subject to the minimization of MD or FA. For the discriminator, $S_0$ denotes the ground truth segmentation, $S_1$ is the result of MD and $S_2$ is for FA.

### Loss

#### Adversarial loss

$$
L_{cGAN}(G,D) = E_{I,S_0}[log D(I,S_0)] + E_I[log(1-D(I,G_1(I)))] + E_I[log(1-D(I,G_2(I)))]
$$

#### Generator consistency loss

This is used to bind the t\wo generators tighter.
$$
L_{GC}(G,D)=\frac{1}{w \cdot h \cdot d}||\phi(I,S)-\phi(I,S_2)||^2_2
$$

#### Data loss

$$
L_{MF1}(G_1, D) = \frac{1}{n}\sum^n_{i=1}(\lambda_1MD_1+FA_1)
$$

$$
L_{MF_2}=\frac{1}{n}\sum^n_{i=1}(MD_2 + \lambda_2FA_2)
$$

$$
L_{MF}(G_1,G_2,D) = L_{MF1}(G_1,D) + L_{MF_2}(G_2,D)
$$



