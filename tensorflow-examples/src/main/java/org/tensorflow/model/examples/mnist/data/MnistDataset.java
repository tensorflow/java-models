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
package org.tensorflow.model.examples.mnist.data;

import static org.tensorflow.tools.ndarray.index.Indices.from;
import static org.tensorflow.tools.ndarray.index.Indices.to;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.tools.ndarray.ByteNdArray;
import org.tensorflow.tools.ndarray.NdArrays;

public class MnistDataset {

  public static final int NUM_CLASSES = 10;

  public static MnistDataset create(int validationSize) {
    try {
      ByteNdArray trainingImages = readArchive(TRAINING_IMAGES_ARCHIVE);
      ByteNdArray trainingLabels = readArchive(TRAINING_LABELS_ARCHIVE);
      ByteNdArray testImages = readArchive(TEST_IMAGES_ARCHIVE);
      ByteNdArray testLabels = readArchive(TEST_LABELS_ARCHIVE);

      if (validationSize > 0) {
        return new MnistDataset(
            trainingImages.slice(from(validationSize)),
            trainingLabels.slice(from(validationSize)),
            trainingImages.slice(to(validationSize)),
            trainingLabels.slice(to(validationSize)),
            testImages,
            testLabels
        );
      }
      return new MnistDataset(trainingImages, trainingLabels, null, null, testImages, testLabels);

    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  public Iterable<ImageBatch> trainingBatches(int batchSize) {
    return () -> new ImageBatchIterator(batchSize, trainingImages, trainingLabels);
  }

  public Iterable<ImageBatch> validationBatches(int batchSize) {
    return () -> new ImageBatchIterator(batchSize, validationImages, validationLabels);
  }

  public Iterable<ImageBatch> testBatches(int batchSize) {
    return () -> new ImageBatchIterator(batchSize, testImages, testLabels);
  }

  public ImageBatch testBatch() {
    return new ImageBatch(testImages, testLabels);
  }

  public long imageSize() {
    return imageSize;
  }

  public long numTrainingExamples() {
    return trainingLabels.shape().size(0);
  }

  public long numTestingExamples() {
    return testLabels.shape().size(0);
  }

  public long numValidationExamples() {
    return validationLabels.shape().size(0);
  }

  private static final String TRAINING_IMAGES_ARCHIVE = "train-images-idx3-ubyte.gz";
  private static final String TRAINING_LABELS_ARCHIVE = "train-labels-idx1-ubyte.gz";
  private static final String TEST_IMAGES_ARCHIVE = "t10k-images-idx3-ubyte.gz";
  private static final String TEST_LABELS_ARCHIVE = "t10k-labels-idx1-ubyte.gz";
  private static final int TYPE_UBYTE = 0x08;

  private final ByteNdArray trainingImages;
  private final ByteNdArray trainingLabels;
  private final ByteNdArray validationImages;
  private final ByteNdArray validationLabels;
  private final ByteNdArray testImages;
  private final ByteNdArray testLabels;
  private final long imageSize;

  private MnistDataset(
      ByteNdArray trainingImages,
      ByteNdArray trainingLabels,
      ByteNdArray validationImages,
      ByteNdArray validationLabels,
      ByteNdArray testImages,
      ByteNdArray testLabels
  ) {
    this.trainingImages = trainingImages;
    this.trainingLabels = trainingLabels;
    this.validationImages = validationImages;
    this.validationLabels = validationLabels;
    this.testImages = testImages;
    this.testLabels = testLabels;
    this.imageSize = trainingImages.get(0).shape().size();
  }

  private static ByteNdArray readArchive(String archiveName) throws IOException {
    DataInputStream archiveStream = new DataInputStream(
        //new GZIPInputStream(new java.io.FileInputStream("src/main/resources/"+archiveName))
        new GZIPInputStream(MnistDataset.class.getClassLoader().getResourceAsStream(archiveName))
    );
    archiveStream.readShort(); // first two bytes are always 0
    byte magic = archiveStream.readByte();
    if (magic != TYPE_UBYTE) {
      throw new IllegalArgumentException("\"" + archiveName + "\" is not a valid archive");
    }
    int numDims = archiveStream.readByte();
    long[] dimSizes = new long[numDims];
    int size = 1;  // for simplicity, we assume that total size does not exceeds Integer.MAX_VALUE
    for (int i = 0; i < dimSizes.length; ++i) {
      dimSizes[i] = archiveStream.readInt();
      size *= dimSizes[i];
    }
    byte[] bytes = new byte[size];
    archiveStream.readFully(bytes);
    return NdArrays.wrap(DataBuffers.of(bytes, true, false), Shape.of(dimSizes));
  }
}
