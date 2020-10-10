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
package org.tensorflow.model.examples.datasets;

import static org.tensorflow.ndarray.index.Indices.range;

import java.util.Iterator;

import org.tensorflow.ndarray.index.Index;
import org.tensorflow.ndarray.ByteNdArray;
import org.tensorflow.ndarray.index.Index;

/** Basic batch iterator across images presented in datset. */
public class ImageBatchIterator implements Iterator<ImageBatch> {

  @Override
  public boolean hasNext() {
    return batchStart < numImages;
  }

  @Override
  public ImageBatch next() {
    long nextBatchSize = Math.min(batchSize, numImages - batchStart);
    Index range = range(batchStart, batchStart + nextBatchSize);
    batchStart += nextBatchSize;
    return new ImageBatch(images.slice(range), labels.slice(range));
  }

  public ImageBatchIterator(int batchSize, ByteNdArray images, ByteNdArray labels) {
    this.batchSize = batchSize;
    this.images = images;
    this.labels = labels;
    this.numImages = images != null ? images.shape().size(0) : 0;
    this.batchStart = 0;
  }

  private final int batchSize;
  private final ByteNdArray images;
  private final ByteNdArray labels;
  private final long numImages;
  private int batchStart;
}
