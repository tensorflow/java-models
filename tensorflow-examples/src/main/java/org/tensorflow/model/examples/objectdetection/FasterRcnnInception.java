package org.tensorflow.model.examples.objectdetection;
/*

download the model from https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1

unzip then untar the model to a local folder - I've used models/faster_rcnn_inception_resnet_v2_1024x1024
From the web page this is the output dictionary
Add some test images into a testimages folder
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

So it appears there's a discrepancy between the web page and running saved_model_cli.
*/

import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
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
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

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


        // get path to model folder (currently in resources
        String modelPath = "models/faster_rcnn_inception_resnet_v2_1024x1024";
        // load saved model
        SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");
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
            long[] shapeArray = imageShape.asArray();

            //The given SavedModel SignatureDef input
            Map<String, Tensor> feed_dict = new HashMap<>();
            feed_dict.put("input_tensor", reshapeTensor(outputImage));
            //The given SavedModel MetaGraphDef key
            Map<String, Tensor> outputTensorMap = model.function("serving_default").call(feed_dict);

            //detection_classes is a model output name
            TFloat32 detectionClasses = (TFloat32) outputTensorMap.get("detection_classes");
            TFloat32 detectionBoxes = (TFloat32) outputTensorMap.get("detection_boxes");
            TFloat32 numDetections = (TFloat32) outputTensorMap.get("num_detections");
            TFloat32 detectionScores = (TFloat32) outputTensorMap.get("detection_scores");

            int numDetects = (int) numDetections.getFloat(0);
            if (numDetects > 0) {
                try {
                    BufferedImage bufferedImage = ImageIO.read(new File(imagePath));
                    Graphics2D graphics2D = bufferedImage.createGraphics();

                    //TODO tf.image.combinedNonMaxSuppression
                    for (int n = 0; n < numDetects; n++) {
                        //put probability and position in outputMap
                        if(detectionScores.getFloat(0,n)> 0.3f) {
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
                            graphics2D.drawString(cocoTreeMap.get(classVal - 1) + " " + classVal, x1, y1);
                        }
                    }
                    //TODO tf.image.encodeJpeg
                    ImageIO.write(bufferedImage, "jpg", new File("image2rcnn.jpg"));
                } catch (IOException e) {

                }
            }
        }
    }

    /**
     * return a 4D tensor from 3D tensor
     *
     * @param tUint8Tensor 3D tensor
     * @return 4D tensor
     */
    private static TUint8 reshapeTensor(TUint8 tUint8Tensor) {
        Ops tf = Ops.create();
        return tf.reshape(tf.constant(tUint8Tensor),
                tf.array(1,
                        tUint8Tensor.shape().asArray()[0],
                        tUint8Tensor.shape().asArray()[1],
                        tUint8Tensor.shape().asArray()[2]
                )
        ).asTensor();
    }

    private static TUint8 drawBoundingBoxes(TUint8 images, TFloat32 boxes, TFloat32 colors ){
        Ops tf = Ops.create();
        Operand<TUint8> imagesOp = tf.constant(images);
        Operand<TFloat32> boxesOp = tf.constant(boxes);
        Operand<TFloat32> colorsOp = tf.constant(colors);
        return tf.image.drawBoundingBoxes(imagesOp,boxesOp,colorsOp).asTensor();
    }

}
