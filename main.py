from face_detection_module import FaceDetection
import cv2

cap = cv2.VideoCapture(0)


face_detection = FaceDetection()

while True:
    success, frame = cap.read()
    if not success:
        print("did not read frame")
        break

    frame, bbox = face_detection.get_faces(frame)
    if len(bbox) > 0:
        print(bbox[0])
        cv2.imshow("image", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
