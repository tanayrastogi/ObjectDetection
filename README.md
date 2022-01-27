# ObjectDetection
OpenCV module for Object Detection using TensorFlow trained model. It uses the cv.dnn module to load the TF trained models. 
The result is a list of detections on the image with label, confidence and bounding box. 

# Models
Currently, the package is tested for two different models,
- Faster RCNN [faster_rcnn_inception_v2_coco_2018_01_28]
- Mask RCNN   [mask-rcnn-coco]

Faster RCNN will return only the label, confidence and bbox for the detection. Mask RCNN will also retrun mask for each detection. 
Check example on the usage. 

### Usage
The packge needs following TF files to run in the folder *ObjectDetection/models/*
-   Proto File      (.pbtxt)
-   Frozen Graph    (.pb)
-   Classes         (.txt)

Please check example usage in **detection.py** on how to use.
All the classes that can be deteted are from the file "Classes.txt". Those are the labels to be mentioned in variable *classes_to_detect* as list. 

### REFERENCE
- [Mask RCNN using OpenCV - PyImageSearch](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)
- [Object Detection - PyImageSearch](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)