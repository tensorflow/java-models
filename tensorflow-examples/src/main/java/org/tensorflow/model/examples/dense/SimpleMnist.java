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
package org.tensorflow.model.examples.dense;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.model.examples.datasets.ImageBatch;
import org.tensorflow.model.examples.datasets.mnist.MnistDataset;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

public class SimpleMnist implements Runnable {
  private static final String TRAINING_IMAGES_ARCHIVE = "mnist/train-images-idx3-ubyte.gz";
  private static final String TRAINING_LABELS_ARCHIVE = "mnist/train-labels-idx1-ubyte.gz";
  private static final String TEST_IMAGES_ARCHIVE = "mnist/t10k-images-idx3-ubyte.gz";
  private static final String TEST_LABELS_ARCHIVE = "mnist/t10k-labels-idx1-ubyte.gz";

  public static void main(String[] args) {
    MnistDataset dataset = MnistDataset.create(VALIDATION_SIZE, TRAINING_IMAGES_ARCHIVE, TRAINING_LABELS_ARCHIVE,
            TEST_IMAGES_ARCHIVE, TEST_LABELS_ARCHIVE);

    try (Graph graph = new Graph()) {
      SimpleMnist mnist = new SimpleMnist(graph, dataset);
      mnist.run();
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);
    
    // Create placeholders and variables, which should fit batches of an unknown number of images
    Placeholder<TFloat32> images = tf.placeholder(TFloat32.class);
    Placeholder<TFloat32> labels = tf.placeholder(TFloat32.class);

    // Create weights with an initial value of 0
    Shape weightShape = Shape.of(dataset.imageSize(), MnistDataset.NUM_CLASSES);
    Variable<TFloat32> weights = tf.variable(tf.zeros(tf.constant(weightShape), TFloat32.class));

    // Create biases with an initial value of 0
    Shape biasShape = Shape.of(MnistDataset.NUM_CLASSES);
    Variable<TFloat32> biases = tf.variable(tf.zeros(tf.constant(biasShape), TFloat32.class));

    // Predict the class of each image in the batch and compute the loss
    Softmax<TFloat32> softmax =
        tf.nn.softmax(
            tf.math.add(
                tf.linalg.matMul(images, weights),
                biases
            )
        );
    Mean<TFloat32> crossEntropy =
        tf.math.mean(
            tf.math.neg(
                tf.reduceSum(
                    tf.math.mul(labels, tf.math.log(softmax)),
                    tf.array(1)
                )
            ),
            tf.array(0)
        );

    // Back-propagate gradients to variables for training
    Optimizer optimizer = new GradientDescent(graph, LEARNING_RATE);
    Op minimize = optimizer.minimize(crossEntropy);

    // Compute the accuracy of the model
    Operand<TInt64> predicted = tf.math.argMax(softmax, tf.constant(1));
    Operand<TInt64> expected = tf.math.argMax(labels, tf.constant(1));
    Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.class), tf.array(0));

    // Run the graph
    try (Session session = new Session(graph)) {

      // Train the model
      for (ImageBatch trainingBatch : dataset.trainingBatches(TRAINING_BATCH_SIZE)) {
        try (TFloat32 batchImages = preprocessImages(trainingBatch.images());
             TFloat32 batchLabels = preprocessLabels(trainingBatch.labels())) {
            session.runner()
                .addTarget(minimize)
                .feed(images.asOutput(), batchImages)
                .feed(labels.asOutput(), batchLabels)
                .run();
        }
      }

      // Test the model
      ImageBatch testBatch = dataset.testBatch();
      try (TFloat32 testImages = preprocessImages(testBatch.images());
           TFloat32 testLabels = preprocessLabels(testBatch.labels());
           TFloat32 accuracyValue = (TFloat32)session.runner()
              .fetch(accuracy)
              .feed(images.asOutput(), testImages)
              .feed(labels.asOutput(), testLabels)
              .run()
              .get(0)) {
        System.out.println("Accuracy: " + accuracyValue.getFloat());
      }
    }
  }

  private static final int VALIDATION_SIZE = 0;
  private static final int TRAINING_BATCH_SIZE = 100;
  private static final float LEARNING_RATE = 0.2f;

  private static TFloat32 preprocessImages(ByteNdArray rawImages) {
    Ops tf = Ops.create();

    // Flatten images in a single dimension and normalize their pixels as floats.
    long imageSize = rawImages.get(0).shape().size();
    return tf.math.div(
        tf.reshape(
            tf.dtypes.cast(tf.constant(rawImages), TFloat32.class),
            tf.array(-1L, imageSize)
        ),
        tf.constant(255.0f)
    ).asTensor();
  }

  private static TFloat32 preprocessLabels(ByteNdArray rawLabels) {
    Ops tf = Ops.create();

    // Map labels to one hot vectors where only the expected predictions as a value of 1.0
    return tf.oneHot(
        tf.constant(rawLabels),
        tf.constant(MnistDataset.NUM_CLASSES),
        tf.constant(1.0f),
        tf.constant(0.0f)
    ).asTensor();
  }

  private final Graph graph;
  private final MnistDataset dataset;
  
  private SimpleMnist(Graph graph, MnistDataset dataset) {
    this.graph = graph;
    this.dataset = dataset;
  }
}
