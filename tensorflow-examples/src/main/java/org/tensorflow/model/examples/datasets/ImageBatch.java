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

import org.tensorflow.ndarray.ByteNdArray;

/** Batch of images for batch training. */
public class ImageBatch {
  
  public ByteNdArray images() {
    return images;
  }
  
  public ByteNdArray labels() {
    return labels;
  }

  public ImageBatch(ByteNdArray images, ByteNdArray labels) {
    this.images = images;
    this.labels = labels;
  }

  private final ByteNdArray images;
  private final ByteNdArray labels;
}
