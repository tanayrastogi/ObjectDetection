# ObjectDetection
OpenCV module for Object Detection using TensorFlow trained model. It uses the cv.dnn module to load the TF trained models. 
The result is a list of detections on the image with label, confidence and bounding box. 

### Usage
The packge needs following TF files to run in the folder *ObjectDetection/models/*
-   Proto File      (.pbtxt)
-   Frozen Graph    (.pb)
-   Classes         (.txt)

Please check **exmaple.py** on how to use.