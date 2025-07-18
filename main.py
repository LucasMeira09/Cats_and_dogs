import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret:
    break

  image_resized = cv2.resize(frame, (640, 640))
  image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

  input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)

  detections = detector(input_tensor)

  boxes = detections["detection_boxes"][0].numpy()
  classes = detections["detection_classes"][0].numpy().astype(int)
  scores = detections["detection_scores"][0].numpy()

  h, w, _ = image_resized.shape
  
  person_detected = False

  for box, score, class_id in zip(boxes, scores, classes):
    if score < 0.5 or class_id != 1: #1 e igual a class pessoa
      continue
    person_detected = True
    ymin, xmin, ymax, xmax = box
    pt1 = (int(xmin * w), int(ymin * h))
    pt2 = (int(xmax * w), int(ymax * h))
    cv2.rectangle(image_resized, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(image_resized, f"{score:.2f}", (pt1[0], pt1[1] - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
  if person_detected:
    cv2.imshow("Detection", image_resized)
  else:
    cv2.imshow("Detection", np.zeros_like(image_resized))

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()