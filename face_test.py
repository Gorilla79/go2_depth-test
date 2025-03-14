import face_recognition
import cv2
import numpy as np
import time

# 카메라 설정
video_capture = cv2.VideoCapture(0)

# 샘플 얼굴 로드 (파일이 없을 경우 예외 처리)
def load_face_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            return encodings[0]
        else:
            print(f"Warning: No face found in {image_path}")
            return None
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# 등록된 얼굴 로드
member1_face_encoding = load_face_encoding("member1.jpg")

# 유효한 얼굴 인코딩만 리스트에 추가
known_face_encodings = []
known_face_names = []

if member1_face_encoding is not None:
    known_face_encodings.append(member1_face_encoding)
    known_face_names.append("Dongsup")

# 얼굴 인식 속도 최적화 변수
face_locations = []
face_encodings = []
face_names = []
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # 성능 최적화를 위해 2프레임마다 한 번씩 얼굴 감지 수행
    frame_count += 1
    if frame_count % 2 == 0:
        continue

    # BGR → RGB 변환
    rgb_frame = frame[:, :, ::-1]

    # 작은 해상도로 다운샘플링 (성능 최적화)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    # 얼굴 검출 (배치 검출로 속도 향상)
    face_locations = face_recognition.batch_face_locations([small_frame])[0]
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # 얼굴 인식 거리 기반 가장 유사한 얼굴 찾기
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # 얼굴 위치를 원래 크기로 변환
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # 얼굴 감지 결과 표시
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
