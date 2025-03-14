import face_recognition
import cv2
import numpy as np
import time

# 📷 카메라 설정
video_capture = cv2.VideoCapture(0)

# ✅ 카메라 성능 향상: 자동 노출 비활성화 + 밝기 조정
video_capture.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
video_capture.set(cv2.CAP_PROP_CONTRAST, 0.5)
video_capture.set(cv2.CAP_PROP_SATURATION, 0.6)
video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 노출 설정
video_capture.set(cv2.CAP_PROP_EXPOSURE, -6)  # 적절한 값 설정 필요

# 🔹 샘플 얼굴 로드 (얼굴이 없을 경우 예외 처리)
def load_face_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image, model="cnn")  # CNN 모델로 더 정확한 감지
        if len(face_locations) == 0:
            print(f"❌ Warning: No face found in {image_path}")
            return None

        # ✅ 인코딩 품질 향상: num_jitters=10 (더 많은 특징 추출)
        encodings = face_recognition.face_encodings(image, known_face_locations=face_locations, num_jitters=10)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return None

# ✅ 등록된 얼굴 로드
member1_face_encoding = load_face_encoding("member1.jpg")

# ✅ 유효한 얼굴 인코딩만 리스트에 추가
known_face_encodings = []
known_face_names = []

if member1_face_encoding is not None:
    known_face_encodings.append(member1_face_encoding)
    known_face_names.append("Dongsup")

# 🏎️ 최적화된 변수
face_locations = []
face_encodings = []
face_names = []
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # ✅ 성능 최적화: 3프레임마다 한 번씩 얼굴 감지 수행 (기존 2 → 3)
    frame_count += 1
    if frame_count % 3 == 0:
        continue

    # 🔹 BGR → RGB 변환
    rgb_frame = frame[:, :, ::-1]

    # ✅ 해상도 줄이기 (속도 최적화)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # 🔹 얼굴 검출 (더 정확한 `cnn` 사용)
    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # ✅ 개선된 얼굴 매칭: 더 낮은 `tolerance=0.5` + 거리 기반 매칭
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # 🔹 원래 해상도로 변환
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # ✅ 얼굴 감지 결과 표시 (더 선명하게)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 결과 화면 출력
    cv2.imshow('Face Recognition', frame)

    # 종료 키 ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
