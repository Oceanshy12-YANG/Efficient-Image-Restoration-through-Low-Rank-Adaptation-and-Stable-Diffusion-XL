# Efficient-Image-Restoration-through-Low-Rank-Adaptation-and-Stable-Diffusion-XL
A Faster and Better Image Restoration Model


![开头照片](https://github.com/user-attachments/assets/0ba3a7aa-1df4-4d7b-96fe-5655f1f9d34d)
In the image shown, we have added blur and SR to the real-world image. It can be seen that the image restored by the model has high quality, but this is thanks to SUPIR. What we have done is to surpass SUPIR in texture and detail, and will have a faster generation speed


## Data set
Large-scale CelebFaces Attributes (CelebA) Dataset, CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. We use 1300 of them as the dataset for face training.




![pip一条线](https://github.com/user-attachments/assets/1d386e0c-c3ee-4929-9910-a529ef34ebae)





## Experiments

### Face
![ppt图片](https://github.com/user-attachments/assets/2235387e-f6a2-4be7-a2df-bac8f84c1cee)

Compare with SUPIR. We apply a mixture of Gaussian blur with σ = 2 and 4× downsampling for super-resolution degradation. Our method has a good restoration effect on facial details, such as scars. For the texture of hair and clothing, our model has a stronger effect than SUPIR.

### Landscape
![MOXINGDUIBI1](https://github.com/user-attachments/assets/f9f9331c-2d83-4ca0-bb4a-0357c74786df)
