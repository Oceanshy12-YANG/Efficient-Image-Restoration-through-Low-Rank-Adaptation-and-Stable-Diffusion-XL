# Efficient-Image-Restoration-through-Low-Rank-Adaptation-and-Stable-Diffusion-XL
A Faster and Better Image Restoration Model(An improved version of the SUPIR model)
## 📖[**Paper**](http://arxiv.org/abs/2408.17060)

[Haiyang Zhao*](https://oceanshy12-YANG.github.io)

![开头照片](https://github.com/user-attachments/assets/0ba3a7aa-1df4-4d7b-96fe-5655f1f9d34d)
In the image shown, we have added blur and SR to the real-world image. It can be seen that the image restored by the model has high quality, but this is thanks to SUPIR. What we have done is to surpass SUPIR in texture and detail, and will have a faster generation speed
![羊毛](https://github.com/user-attachments/assets/e6e113f7-5c0c-4dcf-8016-3f65b1c7b225)
![耳钉](https://github.com/user-attachments/assets/635e2e5e-762e-4a7e-bc3c-d54d4dbcb3b9)
We can see that the texture of the goat's wool on the trained Lora image is more in line with the texture of the original image. In the image of the little girl, we can see that low-quality images basically do not show the earrings. In the SUPIR model, the earrings are also restored to hair, while the image generated by the trained Lora will show the earrings. Therefore, it can be demonstrated that our method generates high fidelity textures.

## Dataset
Large-scale CelebFaces Attributes (CelebA) Dataset, CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. We use 1300 of them as the dataset for face training. Dataset download link: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


## Pipline

![pip一条线](https://github.com/user-attachments/assets/1d386e0c-c3ee-4929-9910-a529ef34ebae)



## 🔥Real-World Applications

### Face
![ppt图片](https://github.com/user-attachments/assets/2235387e-f6a2-4be7-a2df-bac8f84c1cee)

Compare with SUPIR. We apply a mixture of Gaussian blur with σ = 2 and 4× downsampling for super-resolution degradation. Our method has a good restoration effect on facial details, such as scars. For the texture of hair and clothing, our model has a stronger effect than SUPIR.

### Landscape
![MOXINGDUIBI1](https://github.com/user-attachments/assets/f9f9331c-2d83-4ca0-bb4a-0357c74786df)


Moreover, the reduction in computational time does not compromise the quality of the generated images, as evidenced by the consistent performance metrics. Moreover, we can clearly see the differences between the stable SR model and other models, but its performance is not very good. The PASD model performs well in restoring details, such as in case 1. However, PASD has a low ability to restore images with high noise and blur. In case 2, it was unable to restore the windows of distant high-rise buildings and still had noise points in the restored images. In case 3, the restoration of the clock changed its original color.



##  ⚡Model Comparison and Computational Time
 
|  Degradation  |  Method    | PSNR | SSIM   | LPIPS   |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  | 
| Blur ($\sigma=3$) + Noise ($\sigma=30$) | Ours | 32.19 | 0.7434 | 0.0932 | 
|  | Lighting-LoRA | 29.37| 0.5834| 0.1232 | 
|  | HWXL-LoRA | 28.87| 0.6025 | 0.1183 | 
|  | SUPIR | 29.46| 0.4203 | 0.1402 | 
|  | Lighting | 29.63| 0.5523| 0.2085 | 
|  | HWXL |29.13 | 0.5856| 0.1490 | 
|  |  | | |  | 
| SR ($\times 4$) |  Ours | 29.64 | 0.6382 | 0.0916 | 
|  | Lighting-LoRA | 29.03 | 0.5795 | 0.1328 |
|  | HWXL-LoRA |28.78 | 0.6004| 0.1265 |
|  | SUPIR | 29.81 | 0.5934| 0.1432 | 
|  | Lighting | 29.63| 0.5357| 0.2250 | 
|  | HWXL |28.64 | 0.5774| 0.1688|
|  |  | | |  | 
|  Blur ($\sigma=2$) + SR ($\times 4$) | Ours | 29.38 | 0.5651 | 0.1250 |
|  | Lighting-LoRA | 28.11 | 0.5436 | 0.1414 |
|  | HWXL-LoRA | 28.65 | 0.5609 | 0.1293 |
|  | SUPIR | 27.75 | 0.4702 | 0.1306 |
|  | Lighting | 27.33 | 0.4694 | 0.2880 |
|  | HWXL | 28.78 | 0.5495 | 0.1803 |
|  |  | | |  | 
| Blur ($\sigma=2$) + SR ($\times 4$) + Noise ($\sigma=1$)} | Ours | 18.48 | 0.2881 | 0.3505 |
|  | Lighting-LoRA | 17.44 | 0.2590 | 0.4075 |
|  | HWXL-LoRA | 19.70 | 0.2705 | 0.2907 |
|  | SUPIR | 18.57 | 0.2808 | 0.3225 |
|  | Lighting | 17.39 | 0.1681 | 0.6379 |
|  | HWXL | 20.94 | 0.2823 | 0.4472 |
We compared data across different methods under various degradation scenarios using three SDXL models: SDXL, SDXL-lighting, and HelloWorld-XL. The HelloWorld-XL model was trained on a dataset of 20,821 images, which included a diverse range of subjects, including various people, actions, and lifelike animals.

![时间表格对比](https://github.com/user-attachments/assets/f02460bb-e359-4860-8c78-1e6959434f4f)
Our method shows that the LoRA method has improved by nearly 7 seconds compared to before. Compared with the other two models, our method still has the shortest time. 

![对比](https://github.com/user-attachments/assets/f0d08f86-87c8-43ad-8cfd-618930d9c100)
We also conducted tests on low-quality images and compared them with other models, such as DiffBIR, Stable-SR, PASD. We selected the following metrics for quantitative comparison: the  reference metrics PSNR, SSIM, LPIPS.In terms of results, our method achieved the best scores on PSNR and SSIM and LPIPS, indicating that our method has higher perceptual similarity between the restored image and the reference image than other methods.

## 🧩News
Now, I am trying to train my own SDXL and apply lighting technology to achieve acceleration.
