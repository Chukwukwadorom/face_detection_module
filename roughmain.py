import cv2
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1)


cap = cv2.VideoCapture(0)

line_length = 15

while True:
    success, frame = cap.read()
    if not success:
        print("did not read frame")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)


    if results.detections:
        for detection in results.detections:
            # mp_drawing.draw_detection(frame, detection)
            bb_c =  detection.location_data.relative_bounding_box
            score = round(detection.score[0] * 100, 1)
            print(bb_c)
            h, w, c = frame.shape
            xmin, ymin, xw, yh   = int(bb_c.xmin * w), int(bb_c.ymin * h), int(bb_c.width * w), int(bb_c.height *h)
            xmax, ymax = xmin + xw, ymin+yh
            frame = cv2.putText(frame, f'{score}%', (xmin,ymin-10), cv2.FONT_HERSHEY_PLAIN,
                                    2,  (0,215,255), 2, cv2.LINE_AA)


            ##top left
            frame = cv2.line(frame, (xmin, ymin), (xmin+line_length, ymin), (0,215,255), 3)
            frame = cv2.line(frame, (xmin, ymin), (xmin, ymin+line_length), (0,215,255), 3)


            #top right
            frame = cv2.line(frame, (xmax, ymin), (xmax - line_length, ymin), (0,215,255), 3)
            frame = cv2.line(frame, (xmax, ymin), (xmax, ymin + line_length), (0,215,255), 3)

            ##bottom left
            frame = cv2.line(frame, (xmin, ymax), (xmin + line_length, ymax), (0,215,255), 3)
            frame = cv2.line(frame, (xmin, ymax), (xmin, ymax - line_length), (0,215,255), 3)

            ##bottom right
            frame = cv2.line(frame, (xmax, ymax), (xmax - line_length, ymax), (0,215,255), 3)
            frame = cv2.line(frame, (xmax, ymax), (xmax, ymax - line_length), (0,215,255), 3)


            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 215, 255), 1)

    cv2.imshow("image", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
##024961 2,73,97
##683200
#emrald green: R: 31.4, G: 78.4, B: 47.1.


