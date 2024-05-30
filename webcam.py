import cv2
import numpy as np
# import KNNmodel as KNN
import FeatureExtraction as FE
from PIL import ImageFont, ImageDraw, Image
import joblib

# 모델 불러오기
knn = joblib.load('./knn_model.pkl')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 얼굴 인식용 Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 한글 폰트 경로 설정
font_path = "fonts/gulim.ttc"  # Windows의 예시 폰트 경로
font = ImageFont.truetype(font_path, 20)


def compare_faces(face_encoding1, face_encoding2, threshold=0.6):
    """
    Compare two face encodings and determine if they match.
    
    Parameters:
        face_encoding1 (numpy.ndarray): The face encoding of the first face.
        face_encoding2 (numpy.ndarray): The face encoding of the second face.
        threshold (float): Threshold for considering two faces as a match.
        
    Returns:
        bool: True if the faces match, False otherwise.
    """
    # Calculate the Euclidean distance between the two face encodings
    distance = np.linalg.norm(face_encoding1 - face_encoding2)
    
    # Check if the distance is less than the threshold
    if distance < threshold:
        return True  # Faces match
    else:
        return False  # Faces do not match

def get_face_encoding(face_roi):
    # 얼굴 인코딩 계산을 위한 함수
    # 실제로는 얼굴 인코딩을 계산하는 로직을 구현해야 함
    face_img = cv2.resize(face_roi, (100, 100))  # 예시로 얼굴 이미지를 100x100으로 리사이즈
    U, S, VT = np.linalg.svd(face_img.astype(float))  # 특이값 분해 수행
    k = 100
    feature_vector = np.concatenate((S[:k], np.zeros(max(0, k - len(S)))), axis=0)  # 상위 k개의 특잇값 추출
    return feature_vector


def put_text_with_font(image, text, position, font, font_scale=1, font_color=(255, 0, 0)):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=font_color)
    return np.array(img_pil)


while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 검출된 각 얼굴에 대해
    for (x, y, w, h) in faces:
        # 얼굴 부분을 잘라서 인코딩 계산
        face_roi = gray[y:y+h, x:x+w]
        face_encoding = get_face_encoding(face_roi)  # 얼굴 인코딩 계산

        # 얼굴 비교
        face_encoding = face_encoding.reshape(1, -1)  
        name = "인식 불가"

        # KNN 모델을 사용하여 예측
        try:
            label_idx = knn.predict(face_encoding)[0]
            name = FE.team_1_members[label_idx]
        except:
            pass

        # 얼굴에 사각형 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 한글 이름 표시
        frame = put_text_with_font(frame, name, (x, y - 30), font, font_color=(255, 0, 0))

    # 결과 화면 표시
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()