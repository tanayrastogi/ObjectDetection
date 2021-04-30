# Importing the Python packages
import cv2
import ObjectDetection as obj
import time 

# # MODEL parameters
# modelname = "faster_rcnn_inception_v2_coco_2018_01_28"
# proto     = modelname+".pbtxt"
# classes   = "object_detection_classes_coco.txt"
# graph     = "frozen_inference_graph.pb"
# base_confidence=0.6
# classes_to_detect=["person", "car"]

# # MODEL PARAMETERS
modelname = "mask-rcnn-coco"
proto     = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
classes   = "object_detection_classes_coco.txt"
graph     = "frozen_inference_graph.pb"
base_confidence=0.5
classes_to_detect=["person", "car"]

# Model
model = obj.TensorflowModel(modelname, proto, graph, classes,
                               base_confidence, classes_to_detect)

# Load image
image = cv2.imread("image.png")
detections = model.detect(image, "Test Image")

# For showing output
clone = image.copy()
print("\n")
for detect in detections:  
    (startX, startY, endX, endY) = detect["bbox"]
    obj = detect["label"]
    confidence = detect["confidence"]
    
    # Rectangle around the objects detected
    cv2.rectangle(clone, (startX, startY), (endX, endY), (255, 0, 0), 2)
    # But label abnd confidence
    y = startY - 15 if startY - 15 > 15 else startY + 15
    label = "{}: {:.2f}%".format(obj, confidence)
    cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("[RESULT] ", label)
    
# show the output image
time.sleep(0.1)
cv2.imshow("Output", clone)
cv2.waitKey(0)
