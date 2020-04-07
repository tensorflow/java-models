package org.tensorflow.model.examples.mnist.data;

import org.tensorflow.tools.ndarray.ByteNdArray;

public class ImageBatch {
  
  public ByteNdArray images() {
    return images;
  }
  
  public ByteNdArray labels() {
    return labels;
  }

  ImageBatch(ByteNdArray images, ByteNdArray labels) {
    this.images = images;
    this.labels = labels;
  }

  private final ByteNdArray images;
  private final ByteNdArray labels;
}
