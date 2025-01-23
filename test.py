from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("best.pt")
class_names = model.names

cap = cv2.VideoCapture("v1.mp4")  # Initialize webcam capture
count = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:  # Process every 3rd frame for optimization
        continue

    img = cv2.resize(img, (1020, 500))  # Resize the frame
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('img', img)  # Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Use cv2.waitKey(1) for continuous display
        break

cap.release()
cv2.destroyAllWindows()


