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

# HOG 전처리 클래스
class HOGPreprocessor:
    def __init__(self):
        self.win_size = (64, 64)  # HOG를 계산할 윈도우 크기
        self.block_size = (16, 16)  # 블록 크기
        self.block_stride = (8, 8)  # 블록 이동 간격
        self.cell_size = (8, 8)  # 셀 크기
        self.nbins = 9  # 방향(bins) 수
        self.hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.nbins)

    def extract_features(self, image):
        # 이미지를 64x64 크기로 축소
        resized_img = cv2.resize(image, self.win_size)
        # HOG 특징 계산
        hog_feats = self.hog.compute(resized_img)
        return hog_feats.flatten()


# 얼굴 검출 클래스
class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces, gray

# 얼굴 전처리기와 검출기 인스턴스 생성
preprocessor = HOGPreprocessor()
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
                if len(faces) == 0:
                    print("No face detected")
                    continue

                # 얼굴 전처리 및 특징 추출
                for face in faces:
                    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    # 얼굴 부분 이미지 추출
                    face_img = gray[y:y+h, x:x+w]
                    if face_img.size == 0:
                        print(f"Extracted face is empty in {image_filename}. Skipping...")
                        continue

                    hog_features = preprocessor.extract_features(face_img)
                    
                    # 데이터 리스트에 추가
                    X_list.append(hog_features)
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
