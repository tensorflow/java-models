package org.tensorflow.model.examples.mnist;

import java.util.Arrays;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.model.examples.mnist.data.ImageBatch;
import org.tensorflow.model.examples.mnist.data.MnistDataset;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.Softmax;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.ByteNdArray;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

public class SimpleMnist implements Runnable {

  public static void main(String[] args) {
    MnistDataset dataset = MnistDataset.create(VALIDATION_SIZE);
    try (Graph graph = new Graph()) {
      SimpleMnist mnist = new SimpleMnist(graph, dataset);
      mnist.run();
    }
  }

  @Override
  public void run() {
    Ops tf = Ops.create(graph);
    
    // Create placeholders and variables, which should fit batches of an unknown number of images
    Placeholder<TFloat32> images = tf.placeholder(TFloat32.DTYPE);
    Placeholder<TFloat32> labels = tf.placeholder(TFloat32.DTYPE);

    // Create weights with an initial value of 0
    Shape weightShape = Shape.of(dataset.imageSize(), MnistDataset.NUM_CLASSES);
    Variable<TFloat32> weights = tf.variable(weightShape, TFloat32.DTYPE);
    Assign<TFloat32> weightsInit = tf.assign(weights, tf.zerosLike(weights));

    // Create biases with an initial value of 0
    Shape biasShape = Shape.of(MnistDataset.NUM_CLASSES);
    Variable<TFloat32> biases = tf.variable(biasShape, TFloat32.DTYPE);
    Assign<TFloat32> biasesInit = tf.assign(biases, tf.zerosLike(biases));

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
    Gradients gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases));
    Constant<TFloat32> alpha = tf.val(LEARNING_RATE);
    ApplyGradientDescent<TFloat32> weightGradientDescent = tf.train.applyGradientDescent(weights, alpha, gradients.dy(0));
    ApplyGradientDescent<TFloat32> biasGradientDescent = tf.train.applyGradientDescent(biases, alpha, gradients.dy(1));

    // Compute the accuracy of the model
    Operand<TInt64> predicted = tf.math.argMax(softmax, tf.val(1));
    Operand<TInt64> expected = tf.math.argMax(labels, tf.val(1));
    Operand<TFloat32> accuracy = tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), TFloat32.DTYPE), tf.array(0));

    // Run the graph
    try (Session session = new Session(graph)) {

      // Initialize variables
      session.runner()
          .addTarget(weightsInit)
          .addTarget(biasesInit)
          .run();

      // Train the model
      for (ImageBatch trainingBatch : dataset.trainingBatches(TRAINING_BATCH_SIZE)) {
        try (Tensor<TFloat32> batchImages = preprocessImages(trainingBatch.images());
             Tensor<TFloat32> batchLabels = preprocessLabels(trainingBatch.labels())) {
            session.runner()
                .addTarget(weightGradientDescent)
                .addTarget(biasGradientDescent)
                .feed(images.asOutput(), batchImages)
                .feed(labels.asOutput(), batchLabels)
                .run();
        }
      }

      // Test the model
      ImageBatch testBatch = dataset.testBatch();
      try (Tensor<TFloat32> testImages = preprocessImages(testBatch.images());
           Tensor<TFloat32> testLabels = preprocessLabels(testBatch.labels());
           Tensor<TFloat32> accuracyValue = session.runner()
              .fetch(accuracy)
              .feed(images.asOutput(), testImages)
              .feed(labels.asOutput(), testLabels)
              .run()
              .get(0)
              .expect(TFloat32.DTYPE)) {
        System.out.println("Accuracy: " + accuracyValue.data().getFloat());
      }
    }
  }

  private static final int VALIDATION_SIZE = 0;
  private static final int TRAINING_BATCH_SIZE = 100;
  private static final float LEARNING_RATE = 0.2f;

  private static Tensor<TFloat32> preprocessImages(ByteNdArray rawImages) {
    Ops tf = Ops.create();

    // Flatten images in a single dimension and normalize their pixels as floats.
    long imageSize = rawImages.get(0).shape().size();
    return tf.math.div(
        tf.reshape(
            tf.dtypes.cast(tf.val(rawImages), TFloat32.DTYPE),
            tf.array(-1L, imageSize)
        ),
        tf.val(255.0f)
    ).asTensor();
  }

  private static Tensor<TFloat32> preprocessLabels(ByteNdArray rawLabels) {
    Ops tf = Ops.create();

    // Map labels to one hot vectors where only the expected predictions as a value of 1.0
    return tf.oneHot(
        tf.val(rawLabels),
        tf.val(MnistDataset.NUM_CLASSES),
        tf.val(1.0f),
        tf.val(0.0f)
    ).asTensor();
  }

  private Graph graph;
  private MnistDataset dataset;
  
  private SimpleMnist(Graph graph, MnistDataset dataset) {
    this.graph = graph;
    this.dataset = dataset;
  }
}
