# TensorFlow Java Examples

This repository contains examples for [TensorFlow-Java](https://github.com/tensorflow/java).

## Example Models

There are five example models: a LeNet CNN, a VGG CNN, inference using Faster-RCNN, a linear regression and a logistic regression.  

### Faster-RCNN

The Faster-RCNN inference example is in `org.tensorflow.model.examples.cnn.fastrcnn`. 

Download the model from https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1

Unzip then untar the model to a local folder - I've used models/faster_rcnn_inception_resnet_v2_1024x1024.

Create a testimages folder then add some test images into a testimages folder

To run the example add the input image and output image as parameters:

```shell
java -cp target/tensorflow-examples-1.0.0-tfj-1.0.0-rc.2-jar-with-dependencies.jar org.tensorflow.model.examples.cnn.fastrcnn.FasterRcnnInception testimages/image2.jpg image2rcnn.jpg
```

### LeNet CNN

The LeNet example runs on MNIST which is stored in the project's resource directory. It is found in 
`org.tensorflow.model.examples.cnn.lenet`, and can be run with:

```shell
java -cp target/tensorflow-examples-1.0.0-tfj-1.0.0-rc.2-with-dependencies.jar org.tensorflow.model.examples.cnn.lenet.CnnMnist
```

### VGG

The VGG11 example runs on FashionMNIST, stored in the project's resource directory. It is found in 
`org.tensorflow.model.examples.cnn.vgg`, and can be run with:

```shell
java -cp target/tensorflow-examples-1.0.0-tfj-1.0.0-rc.2-with-dependencies.jar org.tensorflow.model.examples.cnn.vgg.VGG11OnFashionMnist
```

### Linear Regression

The linear regression example runs on hard coded data. It is found in `org.tensorflow.model.examples.regression.linear`
and can be run with:

```shell
java -cp target/tensorflow-examples-1.0.0-tfj-1.0.0-rc.2-with-dependencies.jar org.tensorflow.model.examples.regression.linear.LinearRegressionExample
```

### Logistic Regression

The logistic regression example runs on MNIST, stored in the project's resource directory. It is found in 
`org.tensorflow.model.examples.dense.SimpleMnist`, and can be run with:

```shell
java -cp target/tensorflow-examples-1.0.0-tfj-1.0.0-rc.2-with-dependencies.jar org.tensorflow.model.examples.dense.SimpleMnist
```

## Contributions

Contributions of other example models are welcome, for instructions please see the 
[Contributor guidelines](https://github.com/tensorflow/java/blob/master/CONTRIBUTING.md) in TensorFlow-Java.

## Development

This repository tracks TensorFlow-Java and the head will be updated with new releases of TensorFlow-Java.
