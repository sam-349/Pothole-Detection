import cv2

# Replace with the correct URL provided by IP Webcam
ip_camera_url = "http://192.0.0.4:8080"

cap = cv2.VideoCapture(ip_camera_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to fetch frame")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
