# Importing the Python packages
import cv2
import ObjectDetection as obj


# MODEL parameters
modelname = "faster_rcnn_inception_v2_coco_2018_01_28"
proto     = modelname+".pbtxt"
classes   = "object_detection_classes_coco.txt"
graph     = "frozen_inference_graph.pb"
base_confidence=0.6
classes_to_detect=["person", "car"]

# Model
frcnn = obj.TensorflowModel(modelname, proto, graph, classes,
                               base_confidence, classes_to_detect)

# Load image
image = cv2.imread("image.png")
detections = frcnn.detect(image, "Test Image")
for detect in detections:
    print(detect)
    

