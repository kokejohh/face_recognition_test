import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

#app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)
rec_model = app.models['recognition'] # ดึงเฉพาะสมองส่วนจำหน้ามาใช้

# ==========================================
# ฟังก์ชันช่วยแปลงภาพเข้าโมเดล (สำคัญมาก)
# ==========================================
def get_embedding(model, face_image):
    # InsightFace ต้องการภาพขนาด 112x112
    if face_image.shape[0] < 10 or face_image.shape[1] < 10: return None
    face_image = cv2.resize(face_image, (112, 112))
    
    # แปลงสีและโครงสร้างข้อมูลให้ตรงสเปค
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = np.transpose(face_image, (2, 0, 1))
    face_image = np.expand_dims(face_image, axis=0)
    face_image = (face_image.astype(np.float32) - 127.5) / 128.0
    
    # สั่งรันโมเดล
    input_name = model.session.get_inputs()[0].name
    embedding = model.session.run(None, {input_name: face_image})[0]
    return embedding[0]

def compute_sim(feat1, feat2):
    # คำนวณความเหมือน (Cosine Similarity)
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

# ==========================================
# 3. โหลดรูปต้นฉบับ (ใช้แค่รูปเดียว!)
# ==========================================
print("Loading reference face...")
ref_img = cv2.imread("me.jpg") # <--- ใส่ชื่อไฟล์รูปคุณตรงนี้
if ref_img is None: exit()

# ใช้ MediaPipe หาหน้าในรูปต้นฉบับเพื่อตัดมาจำ
h, w, _ = ref_img.shape
results = face_detection.process(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
if not results.detections:
    print("Error: ไม่เจอหน้าในรูปต้นฉบับ (รูปต้องชัด หน้าตรง)")
    exit()

# ตัดรูปหน้า (Crop)
bbox = results.detections[0].location_data.relative_bounding_box
x, y = int(bbox.xmin * w), int(bbox.ymin * h)
w_box, h_box = int(bbox.width * w), int(bbox.height * h)
face_crop = ref_img[max(0,y):y+h_box, max(0,x):x+w_box]

# สร้างรหัสจำหน้า (Embedding) เก็บไว้
ref_embedding = get_embedding(rec_model, face_crop)
print("จำหน้าเสร็จแล้ว! กำลังเปิดกล้อง...")

# ==========================================
# 4. เริ่มกล้อง (Real-time)
# ==========================================
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('http:10.106.23.202:8080/video')

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. ให้ MediaPipe หาหน้าก่อน (เร็ว)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # ดึงตำแหน่งหน้า
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            w_box, h_box = int(bbox.width * w), int(bbox.height * h)
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + w_box), min(h, y + h_box)

            # ตัดภาพหน้าส่งไปเทียบ
            face_crop = frame[y1:y2, x1:x2]
            
            curr_embedding = get_embedding(rec_model, face_crop)
            sim = compute_sim(ref_embedding, curr_embedding)
            name = f"Unknown ({int(sim*100)}%)"
            color = (0, 0, 255) # สีแดง

            if curr_embedding is not None:
                if sim > 0.2:
                    name = f"ME ({int(sim*100)}%)"
                    color = (0, 255, 0) # สีเขียว

            # วาดผลลัพธ์
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('One-Shot Fast Face Rec', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
