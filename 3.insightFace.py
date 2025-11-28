import cv2
import numpy as np
from insightface.app import FaceAnalysis

#app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("Loading reference face...")
img_ref = cv2.imread("me.jpg")
if img_ref is None:
    print("Error: หาไฟล์ me.jpg ไม่เจอ")
    exit()

faces_ref = app.get(img_ref)
if len(faces_ref) == 0:
    print("Error: ไม่พบใบหน้าในรูป me.jpg")
    exit()

ref_embedding = faces_ref[0].normed_embedding
ref_name = "Koke"

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2)

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http://10.106.23.202:8080/video')
print("Start Recognition... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret: break

    faces = app.get(frame)

    for face in faces:
        # ดึงกรอบหน้า (Bounding Box)
        bbox = face.bbox.astype(int)
        
        # ดึงค่า Embedding ของหน้าที่เจอ
        current_embedding = face.normed_embedding
        
        # คำนวณความเหมือนกับหน้าต้นฉบับ
        sim = compute_sim(ref_embedding, current_embedding)
        
        # ตั้งเกณฑ์ (Threshold): ถ้าความเหมือนเกิน 0.5 ให้ถือว่าใช่
        if sim > 0.5:
            text = f"{ref_name} ({sim:.2f})"
            color = (0, 255, 0) # สีเขียว
        else:
            text = f"Unknown ({sim:.2f})"
            color = (0, 0, 255) # สีแดง

        # วาดกรอบและชื่อ
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # InsightFace หาจุดสำคัญบนหน้า (ตา จมูก ปาก) ให้ด้วย 5 จุด
        kps = face.kps.astype(int)
        for kp in kps:
            cv2.circle(frame, (kp[0], kp[1]), 2, (255, 0, 0), -1)

    cv2.imshow('InsightFace Real-time', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
