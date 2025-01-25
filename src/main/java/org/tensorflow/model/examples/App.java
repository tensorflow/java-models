package org.tensorflow.model.examples;

import org.tensorflow.model.examples.cnn.fastrcnn.FasterRcnnInception;
import org.tensorflow.model.examples.cnn.lenet.CnnMnist;
import org.tensorflow.model.examples.cnn.vgg.VGG11OnFashionMnist;
import org.tensorflow.model.examples.dense.SimpleMnist;
import org.tensorflow.model.examples.regression.linear.LinearRegressionExample;
import org.tensorflow.model.examples.tensors.TensorCreation;

import java.util.Arrays;

public class App {

    public static void main(String[] args) {
        switch (args[0]) {
            case "fastrcnn" ->
                FasterRcnnInception.main(Arrays.stream(args, 1, args.length).toArray(String[]::new));
            case "lenet" ->
                CnnMnist.main(Arrays.stream(args, 1, args.length).toArray(String[]::new));
            case "vgg" ->
                VGG11OnFashionMnist.main();
            case "linear" ->
                LinearRegressionExample.main();
            case "logistic" ->
                SimpleMnist.main();
            case "tensors" ->
                TensorCreation.main();
            default ->
                System.out.println("Invalid mode name!");
        }
    }
}
