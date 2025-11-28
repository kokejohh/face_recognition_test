import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
box_spec = mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
keypoint_spec = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://10.106.23.202:4747/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection,
                                      bbox_drawing_spec=box_spec,
                                      keypoint_drawing_spec=keypoint_spec)
            h, w, _ = frame.shape
            box = detection.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)

            cv2.putText(frame, "Koke", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    
    result2 = hands.process(rgb_frame)
    if result2.multi_hand_landmarks:
        for hand_landmarks in result2.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Mediapipe Face Detection", frame)

    # ESC เพื่อออก
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
