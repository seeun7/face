import cv2
import numpy as np
import os
import dlib
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.model_selection import GridSearchCV, train_test_split

# 팀원 이름과 번호 리스트
team_1_members = ['001_PIW', '002_KHH', '003_CSB', '004_KDM', '005_LSE']
team_2_members = ['001_PJS', '002_KMG', '003_KMS', '004_LJH', '005_KUS']
team_3_members = ['001_KHY', '002_KMJ', '003_KKJ', '004_KMS']
team_4_members = ['001_KRE', '002_KYW', '003_LGE', '004_SHS']
team_5_members = ['001_CJY', '002_JUH', '003_MJY', '004_PSG']
team_6_members = ['001_LSC', '002_RYJ', '003_LDW', '004_YSB']
team_7_members = ['001_CYW', '002_JHS', '003_KMJ', '004_PJS', '005_YYS']

# 팀 목록
teams = {
    'Team1': team_1_members,
    'Team2': team_2_members,
    'Team3': team_3_members,
    'Team4': team_4_members,
    'Team5': team_5_members,
    'Team6': team_6_members,
    'Team7': team_7_members
}

dataset_path = 'dataset'  # 데이터셋 디렉토리 경로

# 전처리 클래스
class Preprocessor:
    def __init__(self, k=1800):
        self.k = k

    # 이미지 대비 향상
    def equalize_histogram(self, image):
        
        return cv2.equalizeHist(image)

    # 가우시안 블러
    def apply_gaussian_blur(self, image, kernel_size=(5, 5)): # 커널의 크기는 항상 홀수여야 함
        return cv2.GaussianBlur(image, kernel_size, 0)

    # 밝기 조절
    def adjust_brightness(self, image, alpha=1.2, beta=30):
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
        k = min(self.k, len(S))  # k를 S의 길이와 비교하여 조정
        return np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

    def preprocess(self, face_img):
        face_img_equalized = self.equalize_histogram(face_img)
        face_img_blurred = self.apply_gaussian_blur(face_img_equalized)
        face_img_brightness_adjusted = self.adjust_brightness(face_img_blurred)  
        U, S, VT = self.perform_svd(face_img_brightness_adjusted)
        reconstructed_face = self.reconstruct_image(U, S, VT)
        return reconstructed_face
    
    def flip(self, img):
        flipped_image = self.flip_image(img)
        return flipped_image

# 얼굴 검출 클래스
class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces, gray

# 얼굴 전처리기와 검출기 인스턴스 생성
preprocessor = Preprocessor(k=1800)
detector = FaceDetector()

def main():
    # 전체 팀을 아우르는 label_dict 설정
    label_dict = {}
    label_counter = 0

    # 데이터를 저장할 리스트 초기화
    X_list = []
    y_list = []

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
                    print("No face detected")
                    continue

                # 얼굴 전처리 및 재구성
                for face in faces:
                    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    # 얼굴 부분 이미지 추출
                    face_img = gray[y:y+h, x:x+w]
                    if face_img.size == 0:
                        print(f"Extracted face is empty in {image_filename}. Skipping...")
                        continue

                    reconstructed_face = preprocessor.preprocess(face_img)
                    flip_image = preprocessor.flip(reconstructed_face)
                    
                    # 특이값 분해 수행
                    U, S, VT = preprocessor.perform_svd(flip_image)
                    # 상위 k개의 특잇값을 포함하는 고정된 길이의 벡터로 변환
                    feature_vector = np.concatenate((S[:preprocessor.k], np.zeros(max(0, preprocessor.k - len(S)))), axis=0)

                    # 데이터 리스트에 추가
                    X_list.append(feature_vector)
                    y_list.append(label_dict[member])

    # 데이터를 numpy 배열로 변환
    X = np.array(X_list)
    y = np.array(y_list)

    # label_dict 저장
    with open('label_dict.pkl', 'wb') as file:
        pickle.dump(label_dict, file)

    # 데이터와 레이블을 .pkl 파일로 저장
    with open('data.pkl', 'wb') as file:
        pickle.dump((X, y), file)

    # 데이터와 로드
    with open('data.pkl', 'rb') as file:
        X, y = pickle.load(file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 하이퍼파라미터 그리드 설정
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # GridSearchCV를 사용하여 하이퍼파라미터 튜닝
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼파라미터 출력
    print("Best parameters found: ", grid_search.best_params_)

    # 최적의 모델로 예측 수행
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)

    # 모델 저장
    with open('knn_model.pkl', 'wb') as file:
        pickle.dump(best_knn, file)

if __name__ == '__main__':
    main()