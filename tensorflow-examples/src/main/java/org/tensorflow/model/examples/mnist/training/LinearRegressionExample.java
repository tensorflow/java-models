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
package org.tensorflow.model.examples.mnist.training;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.*;
import org.tensorflow.op.train.ApplyGradientDescent;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * In this example the TensorFlow finds the weight and bias of the linear regression during 1 epoch,
 * training on observations one by one.
 *
 * Also, the weight and bias are extracted and printed.
 */
public class LinearRegressionExample {
    /**
     * Amount of data points.
     */
    private static int N = 10;

    /**
     * This value is used to fill the Y placeholder in prediction.
     */
    private static final float NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER = 2000f;

    public static void main(String[] args) {
        // Prepare the data
        float[] xValues = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
        float[] yValues = new float[10];

        Random rnd = new Random(42);

        for (int i = 0; i < yValues.length; i++) {
            yValues[i] = (float) (10 * xValues[i] + 2 + 0.1 * (rnd.nextDouble() - 0.5));
        }

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Define placeholders
            Placeholder<TFloat32> X = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.scalar()));
            Placeholder<TFloat32> Y = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.scalar()));

            // Define variables
            Variable<TFloat32> weight = tf.variable(Shape.scalar(), TFloat32.DTYPE);
            Variable<TFloat32> bias = tf.variable(Shape.scalar(), TFloat32.DTYPE);

            // Init variables
            Assign<TFloat32> weightInit = tf.assign(weight, tf.constant(1f));
            Assign<TFloat32> biasInit = tf.assign(bias, tf.constant(1f));

            // Define the model function weight*x + bias
            Mul<TFloat32> mul = tf.math.mul(X, weight);
            Add<TFloat32> yPredicted = tf.math.add(mul, bias);

            // Define loss function MSE
            Pow<TFloat32> sum = tf.math.pow(tf.math.sub(yPredicted, Y), tf.constant(2f));
            Div<TFloat32> mse = tf.math.div(sum, tf.constant(2f * N));

            // Define gradient operators
            List<Variable<TFloat32>> variables = new ArrayList<>();
            variables.add(weight);
            variables.add(bias);

            Gradients gradients = tf.gradients(mse, variables);

            Constant<TFloat32> alpha = tf.constant(0.2f);

            ApplyGradientDescent<TFloat32> weightGradientDescent = tf.train.applyGradientDescent(weight, alpha, gradients.dy(0));
            ApplyGradientDescent<TFloat32> biasGradientDescent = tf.train.applyGradientDescent(bias, alpha, gradients.dy(1));

            try (Session session = new Session(graph)) {
                // Initialize graph variables
                session.runner()
                        .addTarget(weightInit)
                        .addTarget(biasInit)
                        .run();

                // Train the model on data
                for (int i = 0; i < xValues.length; i++) {
                    float y = yValues[i];
                    float x = xValues[i];

                    Tensor<TFloat32> xTensor = TFloat32.scalarOf(x);
                    Tensor<TFloat32> yTensor = TFloat32.scalarOf(y);

                    session.runner()
                            .addTarget(weightGradientDescent)
                            .addTarget(biasGradientDescent)
                            .feed(X.asOutput(), xTensor)
                            .feed(Y.asOutput(), yTensor)
                            .run();

                    System.out.println("Training phase");
                    System.out.println("X is " + x + " Y is " + y);
                }

                // Extract the weight value
                Tensor<TFloat32> weightValue = session.runner()
                        .fetch("Variable")
                        .run().get(0).expect(TFloat32.DTYPE);

                System.out.println("Weight is " + weightValue.data().getFloat());

                // Extract the bias value
                Tensor<TFloat32> biasValue = session.runner()
                        .fetch("Variable_1")
                        .run().get(0).expect(TFloat32.DTYPE);

                System.out.println("Bias is " + biasValue.data().getFloat());

                // Let's predict y for x = 10f
                float x = 10f;
                float predictedY = 0f;

                Tensor<TFloat32> xTensor = TFloat32.scalarOf(x);
                Tensor<TFloat32> yTensor = TFloat32.scalarOf(NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER);

                predictedY = session.runner()
                        .feed(X.asOutput(), xTensor)
                        .feed(Y.asOutput(), yTensor)
                        .fetch(yPredicted)
                        .run().get(0).expect(TFloat32.DTYPE).data().getFloat();

                System.out.println("Predicted value: " + predictedY);
            }
        }
    }
}
