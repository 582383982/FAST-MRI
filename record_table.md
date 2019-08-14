# 模型实验结果对比（X4）

|模型|SSIM|PSNR|MSE|NMSE|
|:--:|:--:|:--:|:--:|:--:|
|baseline unet|0.722 +/- 0.2627|31.91 +/- 6.633|9.467e-11 +/- 1.53e-10|0.03432 +/- 0.05023|
|baseline unet adam|<font color=#66b3ff>0.7224 +/- 0.263|<font color=#66b3ff>31.92 +/- 6.643|9.44e-11 +/- 1.522e-10|0.03427 +/- 0.05031|
|dilated unet add left|<font color=#be77ff>0.7238 +/- 0.2631|<font color=#be77ff>31.97 +/- 6.709|9.28e-11 +/- 1.487e-10|0.03401 +/- 0.05027|
|dilated unet cat left|<font color=#66b3ff>0.7223 +/- 0.2625|31.91 +/- 6.641|9.454e-11 +/- 1.533e-10|0.03427 +/- 0.05023|

# 模型实验结果对比（X8）
|模型|SSIM|PSNR|MSE|NMSE|
|:--:|:--:|:--:|:--:|:--:|
|baseline unet|0.6589 +/- 0.2974|30.17 +/- 5.498|1.553e-10 +/- 3.046e-10|0.04679 +/- 0.05715|
|baseline unet adam|<font color=#66b3ff>0.6595 +/- 0.2977|<font color=#66b3ff>30.19 +/- 5.511|1.562e-10 +/- 3.058e-10|0.04665 +/- 0.05727|
|dilated unet add left|<font color=#be77ff>0.6615 +/- 0.2988|<font color=#be77ff>30.28 +/- 5.596|1.518e-10 +/- 2.96e-10|0.04597 +/- 0.05734|
|dilated unet cat left|<font color=#66b3ff>0.6595 +/- 0.2978|<font color=#66b3ff>30.21 +/- 5.529|1.553e-10 +/- 3.046e-10|0.04649 +/- 0.05722|