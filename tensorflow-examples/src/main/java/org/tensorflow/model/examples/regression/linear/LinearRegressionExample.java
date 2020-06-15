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
package org.tensorflow.model.examples.regression.linear;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.*;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

import java.util.List;
import java.util.Random;

/**
 * In this example TensorFlow finds the weight and bias of the linear regression during 1 epoch,
 * training on observations one by one.
 * <p>
 * Also, the weight and bias are extracted and printed.
 */
public class LinearRegressionExample {
    /**
     * Amount of data points.
     */
    private static final int N = 10;

    /**
     * This value is used to fill the Y placeholder in prediction.
     */
    private static final float NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER = 2000f;
    public static final float LEARNING_RATE = 0.1f;
    public static final String WEIGHT_VARIABLE_NAME = "weight";
    public static final String BIAS_VARIABLE_NAME = "bias";

    public static void main(String[] args) {
        // Prepare the data
        float[] xValues = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
        float[] yValues = new float[N];

        Random rnd = new Random(42);

        for (int i = 0; i < yValues.length; i++) {
            yValues[i] = (float) (10 * xValues[i] + 2 + 0.1 * (rnd.nextDouble() - 0.5));
        }

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Define placeholders
            Placeholder<TFloat32> xData = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.scalar()));
            Placeholder<TFloat32> yData = tf.placeholder(TFloat32.DTYPE, Placeholder.shape(Shape.scalar()));

            // Define variables
            Variable<TFloat32> weight = tf.withName(WEIGHT_VARIABLE_NAME).variable(tf.constant(1f));
            Variable<TFloat32> bias = tf.withName(BIAS_VARIABLE_NAME).variable(tf.constant(1f));

            // Define the model function weight*x + bias
            Mul<TFloat32> mul = tf.math.mul(xData, weight);
            Add<TFloat32> yPredicted = tf.math.add(mul, bias);

            // Define loss function MSE
            Pow<TFloat32> sum = tf.math.pow(tf.math.sub(yPredicted, yData), tf.constant(2f));
            Div<TFloat32> mse = tf.math.div(sum, tf.constant(2f * N));

            // Back-propagate gradients to variables for training
            Optimizer optimizer = new GradientDescent(graph, LEARNING_RATE);
            Op minimize = optimizer.minimize(mse);

            try (Session session = new Session(graph)) {
                // Initialize graph variables
                session.run(tf.init());

                // Train the model on data
                for (int i = 0; i < xValues.length; i++) {
                    float y = yValues[i];
                    float x = xValues[i];

                    try (Tensor<TFloat32> xTensor = TFloat32.scalarOf(x);
                         Tensor<TFloat32> yTensor = TFloat32.scalarOf(y)) {

                        session.runner()
                                .addTarget(minimize)
                                .feed(xData.asOutput(), xTensor)
                                .feed(yData.asOutput(), yTensor)
                                .run();

                        System.out.println("Training phase");
                        System.out.println("x is " + x + " y is " + y);
                    }
                }

                // Extract linear regression model weight and bias values
                List<Tensor<?>> tensorList = session.runner()
                        .fetch(WEIGHT_VARIABLE_NAME)
                        .fetch(BIAS_VARIABLE_NAME)
                        .run();

                try (Tensor<TFloat32> weightValue = tensorList.get(0).expect(TFloat32.DTYPE);
                     Tensor<TFloat32> biasValue = tensorList.get(1).expect(TFloat32.DTYPE)) {

                    System.out.println("Weight is " + weightValue.data().getFloat());
                    System.out.println("Bias is " + biasValue.data().getFloat());
                }

                // Let's predict y for x = 10f
                float x = 10f;
                float predictedY = 0f;

                try (Tensor<TFloat32> xTensor = TFloat32.scalarOf(x);
                     Tensor<TFloat32> yTensor = TFloat32.scalarOf(predictedY);
                     Tensor<TFloat32> yPredictedTensor = session.runner()
                             .feed(xData.asOutput(), xTensor)
                             .feed(yData.asOutput(), yTensor)
                             .fetch(yPredicted)
                             .run().get(0).expect(TFloat32.DTYPE)) {

                    predictedY = yPredictedTensor.data().getFloat();

                    System.out.println("Predicted value: " + predictedY);
                }
            }
        }
    }
}
