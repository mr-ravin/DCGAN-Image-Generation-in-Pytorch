### DCGAN implementation in Pytorch

Image Generation using a Deep Convolution Generative Adversarial Network (DCGAN) in Pytorch.

##### In this implementation, 
- Architecture of Generator and Discriminator are slightly different then given in the original paper. Example- for discriminator input, we have normalised image from 0 to 1 instead of -1 to 1. Also we have used `sigmoid activation function` at output the generator instead of `tanh`.
- The generator of DCGAN generates color images with the resolution of 64x64.

##### Directory Architecture:
```
|--dataset/
|   |--train/
|
|--raw_data/
|--generated/
|--models/
|--dataloader.py
|--dcgan_model.py
|--DCGAN in Pytorch.ipynb
|--sample_genereated.jpg
```

### Where is the code ?
The code to train DCGAN and later on generate images from Generator is present inside `DCGAN in Pytorch.ipynb`

### Sample of genereated 100 color images:
`sample_genereated.jpg` showcases 100 color images each with resolution 64x64.

![image](https://github.com/mr-ravin/DCGAN-in-Pytorch/blob/main/sample_generated.jpg?raw=true)

```
Copyright (c) 2023 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
