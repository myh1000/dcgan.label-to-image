# Generative Adversarial Label to Image Synthesis

Label to Image Synthesis essentially is the generation of images based off of an input label. This can be made more complex through the input of multiple labels, each regarding to a seperate feature of the final image, as well as the accuracy and/or quality of the generated image. Having the model converge to a low error rate is difficult as it is highly dependant on the dataset its trained on.


```main.py``` is based off of a standard DCGAN approach with a conditional layer.

```main-p2p.py``` attempts to concat the layers at the start and and keep the old DCGAN format.

To run this code on google cloud compute, go to the [gcloud branch](https://github.com/myh1000/dcgan.label-to-image/tree/gcloud).

Does seem to show worse as the output resolution of the images goes beyond 64px; results below were for 108px output.


## Setup

### Prerequisites
- Python
- numpy
- [TensorFlow](https://www.tensorflow.org/install/) 1.0+

### Training/Testing

```
python main.py train [optional batch_size]
```

```
python main.py test [optional image_output_size]
```


#### Preliminary Results
These were created from a dataset of ~4k images, with labels for each colored hair. 

![](r64.gif)

After ~20 hours

![](r642.gif)


## Acknowledgments

Code borrows heavily from [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) for ```main.py``` and [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow) for ```main-p2p.py```

### References
- [Generating Images from Captions with Attention](https://arxiv.org/abs/1511.02793)
- [IllustrationGAN](https://github.com/tdrussell/IllustrationGAN)
- [https://arxiv.org/pdf/1605.05396.pdf](https://arxiv.org/pdf/1605.05396.pdf)
- [https://arxiv.org/pdf/1605.05395.pdf](https://arxiv.org/pdf/1605.05395.pdf)
- [https://arxiv.org/pdf/1612.03242.pdf](https://arxiv.org/pdf/1612.03242.pdf)
- [Learning What and Where to Draw](http://www.scottreed.info/files/nips2016.pdf)


[http://illustration2vec.net/](http://illustration2vec.net/)

[https://github.com/dragonmeteor/AnimeDrawingsDataset/](https://github.com/dragonmeteor/AnimeDrawingsDataset/) Pose estimation similar to the one described in Learning What and Where to Draw -- MPII Human Pose (MHP).
