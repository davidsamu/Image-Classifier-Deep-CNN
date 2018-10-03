# Image-Classifier-Deep-CNN
Fully configurable Deep Convolutional Neural Network for Image Classification, trained by Backpropagation. 

Implementation does not use of any Deep Learning package (Tensorflow, PyTorch, etc), only Numpy, and the number, order, type and parameters of each layer is fully configurable.

Training results of 4-layer CNN (CONV -> ReLu -> Pool -> Fully Conn) trained to classify 10 classes (road signs) given as 52x52 RGB images, 2000 images per class.

<br/>

<img src="https://github.com/davidsamu/Image-Classifier-Deep-CNN/blob/master/results/4_layers/train.png" width="500">

<br/>

Test results on a different set of images, 100 images per class, showing good generalization performance.

<br/>

<img src="https://github.com/davidsamu/Image-Classifier-Deep-CNN/blob/master/results/4_layers/test.png" width="500">

<br/>
