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

package org.tensorflow.model.examples.objectdetection;
/*

From the web page this is the output dictionary

num_detections: a tf.int tensor with only one value, the number of detections [N].
detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 90] and contains class score distribution (including background) for detection boxes in the image including background class.

However using
venv\Scripts\python.exe venv\Lib\site-packages\tensorflow\python\tools\saved_model_cli.py show --dir models\faster_rcnn_inception_resnet_v2_1024x1024 --all
2021-03-19 12:25:37.000143: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_tensor'] tensor_info:
        dtype: DT_UINT8
        shape: (1, -1, -1, 3)
        name: serving_default_input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['detection_anchor_indices'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300)
        name: StatefulPartitionedCall:0
    outputs['detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 4)
        name: StatefulPartitionedCall:1
    outputs['detection_classes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300)
        name: StatefulPartitionedCall:2
    outputs['detection_multiclass_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 91)
        name: StatefulPartitionedCall:3
    outputs['detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300)
        name: StatefulPartitionedCall:4
    outputs['num_detections'] tensor_info:
        dtype: DT_FLOAT
        shape: (1)
        name: StatefulPartitionedCall:5
    outputs['raw_detection_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 4)
        name: StatefulPartitionedCall:6
    outputs['raw_detection_scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (1, 300, 91)
        name: StatefulPartitionedCall:7
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          input_tensor: TensorSpec(shape=(1, None, None, 3), dtype=tf.uint8, name='input_tensor')

So it appears there's a discrepancy between the web page and running saved_model_cli as
num_detections: a tf.int tensor with only one value, the number of detections [N].
but the actual tensor is DT_FLOAT according to saved_model_cli
also the web page states
detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
but again the actual tensor is DT_FLOAT according to saved_model_cli.
*/


import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;


/**
 * Loads an image using ReadFile and DecodeJpeg and then uses the saved model
 * faster_rcnn/inception_resnet_v2_1024x1024/1 to detect objects with a detection score greater than 0.3
 */

public class FasterRcnnInception {

    private final static String[] cocoLabels = new String[]{
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "street sign",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "hat",
            "backpack",
            "umbrella",
            "shoe",
            "eye glasses",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "plate",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "mirror",
            "dining table",
            "window",
            "desk",
            "toilet",
            "door",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "blender",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
            "hair brush"
    };

    public static void main(String[] params) {


        // get path to model folder
        String modelPath = "models/faster_rcnn_inception_resnet_v2_1024x1024";
        // load saved model
        SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");

        //create a map of the COCO 2017 labels
        TreeMap<Float, String> cocoTreeMap = new TreeMap<>();
        float cocoCount = 0;
        for (String cocoLabel : cocoLabels) {
            cocoTreeMap.put(cocoCount, cocoLabel);
            cocoCount++;
        }
        try (Graph g = new Graph(); Session s = new Session(g)) {

            Ops tf = Ops.create(g);

            //my test image
            String imagePath = "testimages/image2.jpg";

            Constant<TString> fileName = tf.constant(imagePath);

            ReadFile readFile = tf.io.readFile(fileName);

            Session.Runner runner = s.runner();
            s.run(tf.init());

            DecodeJpeg.Options options = DecodeJpeg.channels(3L);
            DecodeJpeg decodeImage = tf.image.decodeJpeg(readFile.contents(), options);

            //fetch image from file
            TUint8 outputImage = (TUint8) runner.fetch(decodeImage).run().get(0);
            Shape imageShape = outputImage.shape();
            //dimensions of test image
            long[] shapeArray = imageShape.asArray();
            //reshape the tensor to 4D for input to model
            Reshape<TUint8> reshape = tf.reshape(tf.constant(outputImage),
                    tf.array(1,
                            outputImage.shape().asArray()[0],
                            outputImage.shape().asArray()[1],
                            outputImage.shape().asArray()[2]
                    )
            );
            TUint8 reshapeTensor = (TUint8) runner.fetch(reshape).run().get(1);
            Map<String, Tensor> feedDict = new HashMap<>();
            //The given SavedModel SignatureDef input
            feedDict.put("input_tensor", reshapeTensor);//reshapeTensor(outputImage));
            //The given SavedModel MetaGraphDef key
            Map<String, Tensor> outputTensorMap = model.function("serving_default").call(feedDict);

            //detection_classes is a model output name
            try (TFloat32 detectionClasses = (TFloat32) outputTensorMap.get("detection_classes");
                 TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes");
                 TFloat32 numDetections = (TFloat32) outputTensorMap.get("num_detections");
                 TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores")) {

                int numDetects = (int) numDetections.getFloat(0);
                if (numDetects > 0) {
                    try {
                        BufferedImage bufferedImage = ImageIO.read(new File(imagePath));
                        Graphics2D graphics2D = bufferedImage.createGraphics();

                        //TODO tf.image.combinedNonMaxSuppression
                        for (int n = 0; n < numDetects; n++) {
                            //put probability and position in outputMap
                            float detectionScore = detectionScores.getFloat(0, n);
                            //only include those classes with detection score greater than 0.3f
                            if (detectionScore > 0.3f) {
                                Float classVal = detectionClasses.getFloat(0, n);
                                //TODO tf.image.drawBoundingBoxes
                                int x1 = (int) (shapeArray[1] * detectionBoxes.getFloat(0, n, 1));
                                int y1 = (int) (shapeArray[0] * detectionBoxes.getFloat(0, n, 0));
                                int x2 = (int) (shapeArray[1] * detectionBoxes.getFloat(0, n, 3));
                                int y2 = (int) (shapeArray[0] * detectionBoxes.getFloat(0, n, 2));
                                graphics2D.setPaint(Color.RED);
                                graphics2D.setStroke(new BasicStroke(5));
                                graphics2D.drawRect(x1, y1, x2 - x1, y2 - y1);
                                graphics2D.setPaint(Color.BLACK);
                                //add a label with percentage score 
                                graphics2D.drawString(cocoTreeMap.get(classVal - 1) + " " +
                                        (new DecimalFormat("#.##").format(detectionScore * 100)) +
                                        "%", x1, y1);
                            }
                        }
                        //TODO tf.image.encodeJpeg
                        ImageIO.write(bufferedImage, "jpg", new File("image2rcnn.jpg"));
                    } catch (IOException e) {
                        System.err.println("Exception with writing image " + e.getMessage());
                    }
                }

            }
            reshapeTensor.close();
            outputImage.close();
        }
    }
}
