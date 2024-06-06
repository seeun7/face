import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 전처리 클래스
class Preprocessor:
    def __init__(self, k=800):
        self.k = k

    # 이미지 대비 향상
    def equalize_histogram(self, image):
        return cv2.equalizeHist(image)

    # 가우시안 블러
    def apply_gaussian_blur(self, image, kernel_size=(5, 5)): # 커널의 크기는 항상 홀수여야 함
        return cv2.GaussianBlur(image, kernel_size, 0)

    # 밝기 조절
    def adjust_brightness(self, image, alpha=3.0, beta=3):
        # alpha는 대비(1.0은 변경 없음), beta는 밝기(0은 변경 없음)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 좌우 반전
    def flip_image(self, image):
        return cv2.flip(image, 1)  # 1은 좌우 반전
    
    # 특이값 분해 수행 후 반환
    def perform_svd(self, image):
        U, S, VT = np.linalg.svd(image.astype(float))
        return U, S, VT

    # 특이값 분해를 사용해 원본의 이미지를 재구성
    def reconstruct_image(self, U, S, VT):
        return np.dot(U[:, :self.k], np.dot(np.diag(S[:self.k]), VT[:self.k, :]))

    def preprocess(self, face_img):
        face_img_equalized = self.equalize_histogram(face_img)
        face_img_blurred = self.apply_gaussian_blur(face_img_equalized)
        face_img_brightness_adjusted = self.adjust_brightness(face_img_blurred, alpha=1.2, beta=30)  
        # 밝기 조절 필요
        U, S, VT = self.perform_svd(face_img_brightness_adjusted)
        reconstructed_face = self.reconstruct_image(U, S, VT)
        return reconstructed_face
    def flip(self,img):
        flipped_image = self.flip_image(img)
        return flipped_image
    
# 얼굴 검출 클래스
class FaceDetector:
    def __init__(self, frontal_cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        self.frontal_cascade = cv2.CascadeClassifier(frontal_cascade_path)
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frontal_faces = self.frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        
        faces = frontal_faces
        return faces, gray




# 팀원 이름과 번호 리스트
team_1_members = ['001_PIW', '002_KHH', '003_CSB', '004_KDM', '005_LSE']
# team_2_members = ['001_AAA', '002_BBB', '003_CCC', '004_DDD']


# 팀 목록
teams = {
    'Team1': team_1_members
    # 'Team2': team_2_members
}

# 얼굴 전처리기와 검출기 인스턴스 생성
preprocessor = Preprocessor(k=800)
detector = FaceDetector()

dataset_path = 'dataset'  # 데이터셋 디렉토리 경로

# 데이터를 저장할 리스트 초기화
X_list = []
y_list = []

# 전체 팀을 아우르는 label_dict 설정
label_dict = {}
label_counter = 0
for team_name, members in teams.items():
    for member in members:
        if member not in label_dict:
            label_dict[member] = label_counter
            label_counter += 1

# 데이터 수집 및 처리
for team_name, members in teams.items():
    team_path = os.path.join(dataset_path, team_name)  # 각 팀의 디렉토리 경로
    for member in members:
        member_id = member.split('_')[0]
        member_name = member.split('_')[1]
        member_path = os.path.join(team_path, member)  # 각 멤버의 디렉토리 경로
        for photo_num in range(1, 101):  # 사진의 개수는 조정
            image_filename = f'{team_name}_{member_id}_{member_name}_{photo_num:04d}.jpg'  # 이미지 파일 이름
            image_path = os.path.join(member_path, image_filename)  # 이미지 파일 경로

            # 경로 출력 및 존재 여부 확인
            print(f"Checking: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {image_path} not found.")
                continue

            faces, gray = detector.detect_faces(image)
            member_display_name = member.split('_')[1]

            if len(faces) == 0:
                # 얼굴이 인식되지 않으면 테두리 없는 사진을 출력
                print("no face")
                continue

            # 얼굴 전처리 및 재구성
            for (x, y, w, h) in faces:
                # 얼굴 부분 이미지 추출
                face_img = gray[y:y+h, x:x+w]
                
                reconstructed_face = preprocessor.preprocess(face_img)
                flip_image=preprocessor.flip(reconstructed_face)
                # 특이값 분해 수행
                #U, S, VT = preprocessor.perform_svd(flip_image)
                U, S, VT = preprocessor.perform_svd(reconstructed_face)
                # 상위 k개의 특잇값을 포함하는 고정된 길이의 벡터로 변환
                feature_vector = np.concatenate((S[:preprocessor.k], np.zeros(max(0, preprocessor.k - len(S)))), axis=0)

                # 데이터 리스트에 추가
                X_list.append(feature_vector)
                y_list.append(label_dict[member])

# 데이터를 numpy 배열로 변환
X = np.array(X_list)
y = np.array(y_list)
    
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X,y)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, gray = detector.detect_faces(frame)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # Preprocessor 클래스를 사용하여 전처리 수행
        reconstructed_face = preprocessor.preprocess(face_img)

        # 상위 k개의 특잇값을 포함하는 고정된 길이의 벡터로 변환
        U, S, VT = preprocessor.perform_svd(reconstructed_face)
        feature_vector = np.concatenate((S[:preprocessor.k], np.zeros(max(0, preprocessor.k - len(S)))), axis=0)
        feature_vector = feature_vector.reshape(1, -1)  # KNN 입력 형식에 맞게 변형

        # 예측 수행
        predicted_label = knn.predict(feature_vector)[0]
        if knn.predict_proba(feature_vector).max() > 0.7:  #임계값
            predicted_name = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]
        else:
            predicted_name = "unknown"

        # 예측된 이름을 화면에 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
