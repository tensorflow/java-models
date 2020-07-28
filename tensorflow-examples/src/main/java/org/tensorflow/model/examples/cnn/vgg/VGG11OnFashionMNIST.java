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

import org.tensorflow.model.examples.datasets.mnist.MnistDataset;

import java.util.logging.Logger;

/**
 * Trains and evaluates VGG'11 model on FashionMNIST dataset.
 */
public class VGG11OnFashionMNIST {
    // Hyper-parameters
    public static final int EPOCHS = 1;
    public static final int BATCH_SIZE = 500;

    // Fashion MNIST dataset paths
    public static final String TRAINING_IMAGES_ARCHIVE = "fashionmnist/train-images-idx3-ubyte.gz";
    public static final String TRAINING_LABELS_ARCHIVE = "fashionmnist/train-labels-idx1-ubyte.gz";
    public static final String TEST_IMAGES_ARCHIVE = "fashionmnist/t10k-images-idx3-ubyte.gz";
    public static final String TEST_LABELS_ARCHIVE = "fashionmnist/t10k-labels-idx1-ubyte.gz";

    private static final Logger logger = Logger.getLogger(VGG11OnFashionMNIST.class.getName());

    public static void main(String[] args) {
        logger.info("Data loading.");
        MnistDataset dataset = MnistDataset.create(0, TRAINING_IMAGES_ARCHIVE, TRAINING_LABELS_ARCHIVE, TEST_IMAGES_ARCHIVE, TEST_LABELS_ARCHIVE);

        try (VGGModel vggModel = new VGGModel()) {
            logger.info("Model training.");
            vggModel.train(dataset, EPOCHS, BATCH_SIZE);

            logger.info("Model evaluation.");
            vggModel.test(dataset, BATCH_SIZE);
        }
    }
}
