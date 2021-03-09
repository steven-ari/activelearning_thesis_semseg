# activelearning_thesis_semseg

Reducing demand for training model for semantic segmentation using Active Learning. A great literature to start understanding [Active Learning](http://www.cs.cmu.edu/~bsettles/pub/settles.activelearning.pdf)

## Result

1. Model: VGG19 as Encoder + U-Net as Classifier: 

In this thesis, two variant of active learning methods were applied (blue and orange), and both shows improvement in information gain on the same amount of training sample allowed. Each datapoint and its whisker represents 5 experiments. The red line shows the changes of accuracy related to changes of number of training sample for random sample selection setting.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src=image_examples/bulk_mini_all.png width="50%">.


2. Model: DeeplabV3 [*(link)*](https://paperswithcode.com/method/deeplabv3#:~:text=DeepLabv3%20is%20a%20semantic%20segmentation,by%20adopting%20multiple%20atrous%20rates.)

Comparison of QBC vs without AL on different sample sizes. Each datapoint and its whisker represents 5 experiments. The red line shows the changes of accuracy related to changes of number of training sample for random sample selection setting. Noticable difference with previous result: Since DeeplabV3 is a larger model, it is therefore a more challenging one to reduce the demand for training data.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src=image_examples/bulk_all.png width="50%">.

### Inference Sample 
Average IOU: ± 0.65 IOU using DeeplabV3 [*(link)*](https://paperswithcode.com/method/deeplabv3#:~:text=DeepLabv3%20is%20a%20semantic%20segmentation,by%20adopting%20multiple%20atrous%20rates.). The work that popularized Atrous Spatial Pyramid Pooling [*(link)*](https://arxiv.org/abs/1606.00915v2) on semantic segmentation.

<img src=image_examples/inference_example.png width="60%">.

## Dataset 

#### MNIST
Image recognition on Handwritten digits [*(link)*](http://yann.lecun.com/exdb/mnist/)

<img src=image_examples/mnist_example.png width="30%">.

#### Cityscapes
Traffic sceneries taken in Germany [*(link)*](https://www.cityscapes-dataset.com/)

<img src=image_examples/cityscapes_example.png width="30%">.

## Sources

Thesis: [link](https://drive.google.com/file/d/1jBDupzZuklW6y4vR5nasu3iL14DSzlUu/view?usp=sharing)

Sources and contributions can be found in the thesis.
