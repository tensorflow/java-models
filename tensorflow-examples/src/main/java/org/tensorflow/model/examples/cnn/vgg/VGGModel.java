/*
 *  Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  =======================================================================
 */
package org.tensorflow.model.examples.cnn.vgg;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.model.examples.datasets.ImageBatch;
import org.tensorflow.model.examples.datasets.mnist.MnistDataset;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.OneHot;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Conv2d;
import org.tensorflow.op.nn.MaxPool;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.op.nn.SoftmaxCrossEntropyWithLogits;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;

/**
 * Describes the VGGModel.
 */
public class VGGModel implements AutoCloseable {
    private static final int PIXEL_DEPTH = 255;
    private static final int NUM_CHANNELS = 1;
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_LABELS = MnistDataset.NUM_CLASSES;
    private static final long SEED = 123456789L;

    private static final String PADDING_TYPE = "SAME";
    public static final String INPUT_NAME = "input";
    public static final String OUTPUT_NAME = "output";
    public static final String TARGET = "target";
    public static final String TRAIN = "train";
    public static final String TRAINING_LOSS = "training_loss";

    private static final Logger logger = Logger.getLogger(VGGModel.class.getName());

    private final Graph graph;

    private final Session session;

    public VGGModel() {
        graph = compile();
        session = new Session(graph);
    }

    public static Graph compile() {
        Graph graph = new Graph();

        Ops tf = Ops.create(graph);

        // Inputs
        Placeholder<TUint8> input = tf.withName(INPUT_NAME).placeholder(TUint8.class,
                Placeholder.shape(Shape.of(-1, IMAGE_SIZE, IMAGE_SIZE)));
        Reshape<TUint8> input_reshaped = tf
                .reshape(input, tf.array(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS));
        Placeholder<TUint8> labels = tf.withName(TARGET).placeholder(TUint8.class);

        // Scaling the features
        Constant<TFloat32> centeringFactor = tf.constant(PIXEL_DEPTH / 2.0f);
        Constant<TFloat32> scalingFactor = tf.constant((float) PIXEL_DEPTH);
        Operand<TFloat32> scaledInput = tf.math
                .div(tf.math.sub(tf.dtypes.cast(input_reshaped, TFloat32.class), centeringFactor),
                        scalingFactor);

        Relu<TFloat32> relu1 = vggConv2DLayer("1", tf, scaledInput, new int[]{3, 3, NUM_CHANNELS, 32}, 32);

        MaxPool<TFloat32> pool1 = vggMaxPool(tf, relu1);

        Relu<TFloat32> relu2 = vggConv2DLayer("2", tf, pool1, new int[]{3, 3, 32, 64}, 64);

        MaxPool<TFloat32> pool2 = vggMaxPool(tf, relu2);

        Relu<TFloat32> relu3 = vggConv2DLayer("3", tf, pool2, new int[]{3, 3, 64, 128}, 128);
        Relu<TFloat32> relu4 = vggConv2DLayer("4", tf, relu3, new int[]{3, 3, 128, 128}, 128);

        MaxPool<TFloat32> pool3 = vggMaxPool(tf, relu4);

        Relu<TFloat32> relu5 = vggConv2DLayer("5", tf, pool3, new int[]{3, 3, 128, 256}, 256);
        Relu<TFloat32> relu6 = vggConv2DLayer("6", tf, relu5, new int[]{3, 3, 256, 256}, 256);

        MaxPool<TFloat32> pool4 = vggMaxPool(tf, relu6);

        Relu<TFloat32> relu7 = vggConv2DLayer("7", tf, pool4, new int[]{3, 3, 256, 256}, 256);
        Relu<TFloat32> relu8 = vggConv2DLayer("8", tf, relu7, new int[]{3, 3, 256, 256}, 256);

        MaxPool<TFloat32> pool5 = vggMaxPool(tf, relu8);

        Reshape<TFloat32> flatten = vggFlatten(tf, pool5);

        Add<TFloat32> loss = buildFCLayersAndRegularization(tf, labels, flatten);

        Optimizer optimizer = new Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f);

        optimizer.minimize(loss, TRAIN);

        return graph;
    }

    public static Add<TFloat32> buildFCLayersAndRegularization(Ops tf, Placeholder<TUint8> labels, Reshape<TFloat32> flatten) {
        int fcBiasShape = 100;
        int[] fcWeightShape = {256, fcBiasShape};

        Variable<TFloat32> fc1Weights = tf.variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(fcWeightShape), TFloat32.class,
                        TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc1Biases = tf
                .variable(tf.fill(tf.array(new int[]{fcBiasShape}), tf.constant(0.1f)));
        Relu<TFloat32> fcRelu = tf.nn
                .relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases));

        // Softmax layer
        Variable<TFloat32> fc2Weights = tf.variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(fcBiasShape, NUM_LABELS), TFloat32.class,
                        TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Variable<TFloat32> fc2Biases = tf
                .variable(tf.fill(tf.array(new int[]{NUM_LABELS}), tf.constant(0.1f)));

        Add<TFloat32> logits = tf.math.add(tf.linalg.matMul(fcRelu, fc2Weights), fc2Biases);

        // Predicted outputs
        tf.withName(OUTPUT_NAME).nn.softmax(logits);

        // Loss function & regularization
        OneHot<TFloat32> oneHot = tf
                .oneHot(labels, tf.constant(10), tf.constant(1.0f), tf.constant(0.0f));
        SoftmaxCrossEntropyWithLogits<TFloat32> batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, oneHot);
        Mean<TFloat32> labelLoss = tf.math.mean(batchLoss.loss(), tf.constant(0));
        Add<TFloat32> regularizers = tf.math.add(tf.nn.l2Loss(fc1Weights), tf.math
                .add(tf.nn.l2Loss(fc1Biases),
                        tf.math.add(tf.nn.l2Loss(fc2Weights), tf.nn.l2Loss(fc2Biases))));
        return tf.withName(TRAINING_LOSS).math
                .add(labelLoss, tf.math.mul(regularizers, tf.constant(5e-4f)));
    }

    public static Reshape<TFloat32> vggFlatten(Ops tf, MaxPool<TFloat32> pool2) {
        return tf.reshape(pool2, tf.concat(Arrays
                .asList(tf.slice(tf.shape(pool2), tf.array(new int[]{0}), tf.array(new int[]{1})),
                        tf.array(new int[]{-1})), tf.constant(0)));
    }

    public static MaxPool<TFloat32> vggMaxPool(Ops tf, Relu<TFloat32> relu1) {
        return tf.nn
                .maxPool(relu1, tf.array(1, 2, 2, 1), tf.array(1, 2, 2, 1),
                        PADDING_TYPE);
    }

    public static Relu<TFloat32> vggConv2DLayer(String layerName, Ops tf, Operand<TFloat32> scaledInput, int[] convWeightsL1Shape, int convBiasL1Shape) {
        Variable<TFloat32> conv1Weights = tf.withName("conv2d_" + layerName).variable(tf.math.mul(tf.random
                .truncatedNormal(tf.array(convWeightsL1Shape), TFloat32.class,
                        TruncatedNormal.seed(SEED)), tf.constant(0.1f)));
        Conv2d<TFloat32> conv = tf.nn
                .conv2d(scaledInput, conv1Weights, Arrays.asList(1L, 1L, 1L, 1L), PADDING_TYPE);
        Variable<TFloat32> convBias = tf
                .withName("bias2d_" + layerName).variable(tf.fill(tf.array(new int[]{convBiasL1Shape}), tf.constant(0.0f)));
        return tf.nn.relu(tf.withName("biasAdd_" + layerName).nn.biasAdd(conv, convBias));
    }

    public void train(MnistDataset dataset, int epochs, int minibatchSize) {
        int interval = 0;
        // Train the model
        for (int i = 0; i < epochs; i++) {
            for (ImageBatch trainingBatch : dataset.trainingBatches(minibatchSize)) {
                try (TUint8 batchImages = TUint8.tensorOf(trainingBatch.images());
                     TUint8 batchLabels = TUint8.tensorOf(trainingBatch.labels());
                     TFloat32 loss = (TFloat32)session.runner()
                             .feed(TARGET, batchLabels)
                             .feed(INPUT_NAME, batchImages)
                             .addTarget(TRAIN)
                             .fetch(TRAINING_LOSS)
                             .run().get(0)) {

                    logger.log(Level.INFO,
                            "Iteration = " + interval + ", training loss = " + loss.getFloat());

                }
                interval++;
            }
        }
    }

    public void test(MnistDataset dataset, int minibatchSize) {
        int correctCount = 0;
        int[][] confusionMatrix = new int[10][10];

        for (ImageBatch trainingBatch : dataset.testBatches(minibatchSize)) {
            try (TUint8 transformedInput = TUint8.tensorOf(trainingBatch.images());
                 TFloat32 outputTensor = (TFloat32)session.runner()
                         .feed(INPUT_NAME, transformedInput)
                         .fetch(OUTPUT_NAME).run().get(0)) {

                ByteNdArray labelBatch = trainingBatch.labels();
                for (int k = 0; k < labelBatch.shape().size(0); k++) {
                    byte trueLabel = labelBatch.getByte(k);
                    int predLabel;

                    predLabel = argmax(outputTensor.slice(Indices.at(k), Indices.all()));
                    if (predLabel == trueLabel) {
                        correctCount++;
                    }

                    confusionMatrix[trueLabel][predLabel]++;
                }
            }
        }

        logger.info("Final accuracy = " + ((float) correctCount) / dataset.numTestingExamples());

        StringBuilder sb = new StringBuilder();
        sb.append("Label");
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%1$5s", "" + i));
        }
        sb.append("\n");

        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%1$5s", "" + i));
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                sb.append(String.format("%1$5s", "" + confusionMatrix[i][j]));
            }
            sb.append("\n");
        }

        System.out.println(sb);
    }

    /**
     * Find the maximum probability and return it's index.
     *
     * @param probabilities The probabilites.
     * @return The index of the max.
     */
    public static int argmax(FloatNdArray probabilities) {
        float maxVal = Float.NEGATIVE_INFINITY;
        int idx = 0;
        for (int i = 0; i < probabilities.shape().size(0); i++) {
            float curVal = probabilities.getFloat(i);
            if (curVal > maxVal) {
                maxVal = curVal;
                idx = i;
            }
        }
        return idx;
    }

    @Override
    public void close() {
        session.close();
        graph.close();
    }
}
