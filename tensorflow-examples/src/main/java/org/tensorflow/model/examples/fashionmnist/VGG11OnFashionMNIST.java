/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.model.examples.fashionmnist;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.optimizers.*;
import org.tensorflow.model.examples.mnist.data.ImageBatch;
import org.tensorflow.model.examples.mnist.data.MnistDataset;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.*;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.ByteNdArray;
import org.tensorflow.tools.ndarray.FloatNdArray;
import org.tensorflow.tools.ndarray.index.Indices;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Builds a VGG'11 model for MNIST.
 */
public class VGG11OnFashionMNIST {
    // Hyper-parameters
    public static final int EPOCHS = 2;

    public static final int BATCH_SIZE = 500;

    private static final Logger logger = Logger.getLogger(VGG11OnFashionMNIST.class.getName());

    public static void main(String[] args) {
        logger.info("Data loading.");
        FashionMnistDataset dataset = FashionMnistDataset.create(0);

        try(VGGModel vggModel = new VGGModel()) {
            logger.info("Model training.");
            vggModel.train(dataset, EPOCHS, BATCH_SIZE);

            logger.info("Model evaluation.");
            vggModel.test(dataset, BATCH_SIZE);
        }
    }
}
