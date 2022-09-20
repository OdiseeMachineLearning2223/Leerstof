import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

print('loading model...')
hub_model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_1024x1024/1")
print('model loaded!')

cap = cv2.VideoCapture(0)
width = 640
height = 480
cap.set(3,width) # adjust width
cap.set(4,height) # adjust height

print("\nStart webcam feed")

labels = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

plt.ion()
# show webcam feed
while True:

    print("Start new frame")

    success, img = cap.read()
    plt.clf()
    plt.imshow(img)
    
    # execute object detection
    img_reshaped = img.reshape((1, height, width, 3))
    detector_output = hub_model(img_reshaped)
    
    # draw output on image
    label_id_offset = 0
    image_np_with_detections = img_reshaped.copy()

    boxes = np.squeeze(detector_output['detection_boxes'])
    scores = np.squeeze(detector_output['detection_scores'])
    classes = np.squeeze(detector_output['detection_classes'])

    _, im_height, im_width,_ = image_np_with_detections.shape
    for idx, box in enumerate(boxes):
        if(scores[idx] > 0.4):
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin * im_height), int(xmin * im_width), int(ymax * im_height), int(xmax * im_width)
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
            
            plt.gca().add_patch(rect)
            
            print(classes[idx])
            try:
                plt.text(xmin+15, ymin+15, labels[int(classes[idx])-1] + " " + str(int(scores[idx]*100)) + "%")
            except:
                pass

    plt.pause(0.01)