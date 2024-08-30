# Efficient-Image-Restoration-through-Low-Rank-Adaptation-and-Stable-Diffusion-XL
A Faster and Better Image Restoration Model
## 📖[**Paper**](https://arxiv.org/submit/5822392/view)

[Haiyang Zhao*](https://oceanshy12-YANG.github.io)

![开头照片](https://github.com/user-attachments/assets/0ba3a7aa-1df4-4d7b-96fe-5655f1f9d34d)
In the image shown, we have added blur and SR to the real-world image. It can be seen that the image restored by the model has high quality, but this is thanks to SUPIR. What we have done is to surpass SUPIR in texture and detail, and will have a faster generation speed
![羊毛](https://github.com/user-attachments/assets/e6e113f7-5c0c-4dcf-8016-3f65b1c7b225)
![耳钉](https://github.com/user-attachments/assets/635e2e5e-762e-4a7e-bc3c-d54d4dbcb3b9)


## Dataset
Large-scale CelebFaces Attributes (CelebA) Dataset, CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. We use 1300 of them as the dataset for face training.


## Pipline

![pip一条线](https://github.com/user-attachments/assets/1d386e0c-c3ee-4929-9910-a529ef34ebae)





## 🔥Real-World Applications

### Face
![ppt图片](https://github.com/user-attachments/assets/2235387e-f6a2-4be7-a2df-bac8f84c1cee)

Compare with SUPIR. We apply a mixture of Gaussian blur with σ = 2 and 4× downsampling for super-resolution degradation. Our method has a good restoration effect on facial details, such as scars. For the texture of hair and clothing, our model has a stronger effect than SUPIR.

### Landscape
![MOXINGDUIBI1](https://github.com/user-attachments/assets/f9f9331c-2d83-4ca0-bb4a-0357c74786df)

We also conducted tests on low-quality images and compared them with other models, such as DiffBIR, Stable-SR, PASD. We selected the following metrics for quantitative comparison: the  reference metrics PSNR, SSIM, LPIPS.In terms of results, our method achieved the best scores on PSNR and SSIM and LPIPS, indicating that our method has higher perceptual similarity between the restored image and the reference image than other methods.

LoRA reduces the complexity of the model through parameter decomposition, thereby reducing time. As shown in tab:3, the comparison between the original method and our method shows that the LoRA method has improved by nearly 7 seconds compared to before. Compared with the other two models, our method still has the shortest time. However, StableSR requires 200 steps to generate a perfect image and consumes a lot of time. This efficiency gain demonstrates the effectiveness of our approach in handling large-scale models. Moreover, the reduction in computational time does not compromise the quality of the generated images, as evidenced by the consistent performance metrics. Moreover, fig:4, we can clearly see the differences between the stable SR model and other models, but its performance is not very good. The PASD model performs well in restoring details, such as in case 1. However, PASD has a low ability to restore images with high noise and blur. In case 2, it was unable to restore the windows of distant high-rise buildings and still had noise points in the restored images. In case 3, the restoration of the clock changed its original color.
