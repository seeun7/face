import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# 전처리 클래스
class Preprocessor:
    def __init__(self, k=100):
        self.k = k

    def equalize_histogram(self, image):
        return cv2.equalizeHist(image)

    def apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    def perform_svd(self, image):
        U, S, VT = np.linalg.svd(image.astype(float))
        return U, S, VT

    def reconstruct_image(self, U, S, VT):
        return np.dot(U[:, :self.k], np.dot(np.diag(S[:self.k]), VT[:self.k, :]))

    def preprocess(self, face_img):
        face_img_equalized = self.equalize_histogram(face_img)
        face_img_blurred = self.apply_gaussian_blur(face_img_equalized)
        U, S, VT = self.perform_svd(face_img_blurred)
        reconstructed_face = self.reconstruct_image(U, S, VT)
        return face_img_equalized, reconstructed_face

# 얼굴 검출 클래스
class FaceDetector:
    def __init__(self, frontal_cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                       profile_cascade_path=cv2.data.haarcascades + 'haarcascade_profileface.xml'):
        self.frontal_cascade = cv2.CascadeClassifier(frontal_cascade_path)
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # scaleFactor와 minNeighbors 파라미터 조정
        frontal_faces = self.frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        profile_faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        faces = list(frontal_faces) + list(profile_faces)  # 정면 얼굴과 측면 얼굴을 통합
        return faces, gray

# 팀원 이름과 번호 리스트
team_1_members = ['001_PIW', '002_KHH', '003_CSB', '004_KDM', '005_LSE']

# 팀 목록
teams = {'Team1': team_1_members}

datapath = 'dataset'
X_list = []
y_list = []
face_list = []
face_rcg = FaceDetector()  # 첫 번째 코드의 FaceDetector 클래스 사용
label_dict = {member: idx for idx, member in enumerate(team_1_members)}
IMG_SIZE = 200  # 이미지 크기

def preprocess(n_components):
    preprocessor = Preprocessor(k=n_components)  # 첫 번째 코드의 Preprocessor 클래스 사용
    all_faces = []
    for t_num, members in teams.items():
        folder = os.path.join(datapath, t_num)
        
        for member in members:
            m_id = member.split('_')[0]
            m_name = member.split('_')[1]
            m_fold = os.path.join(folder, member)
            
            for pic in range(1, 101):
                filename = f'{t_num}_{m_id}_{m_name}_{pic:04d}.jpg'
                img_path = os.path.join(m_fold, filename)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Image {img_path} not found!")
                    continue
                
                faces, gray = face_rcg.detect_faces(image)
                
                for (x, y, w, h) in faces:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face_eq, face_recon = preprocessor.preprocess(face)  # 전처리 수행
                    all_faces.append(face_recon)
                    y_list.append(label_dict[member])
    
    all_faces = np.array(all_faces)
    mean_face = np.mean(all_faces, axis=0)
    centerdata = all_faces - mean_face
    
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(centerdata.reshape(centerdata.shape[0], -1))
    
    return pca, transformed_data

n_components = 100  # 이 값을 조정하여 최적의 성능을 찾습니다
pca, X_list = preprocess(n_components)

def KNN_train(X_, y_, n=10):
    k_model = KNeighborsClassifier(n_neighbors=n)
    k_model.fit(X_, y_)
    return k_model

# 데이터 세트를 훈련 세트와 테스트 세트로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X_list, y_list, test_size=0.3, random_state=42)

# KNN 모델을 훈련합니다.
km = KNN_train(X_train, y_train)

# 테스트 데이터에 대해 예측을 수행합니다.
y_pred = km.predict(X_test)

# 정확도를 계산합니다.
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 정밀도와 재현율을 계산합니다.
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# 혼동 행렬 생성
cm = confusion_matrix(y_test, y_pred)

# 실제 클래스 이름을 가져오기 위해 역으로 매핑
inverse_label_dict = {v: k for k, v in label_dict.items()}

# 클래스 이름을 정렬된 순서로 가져오기
class_names = [inverse_label_dict[i] for i in sorted(inverse_label_dict.keys())]

# True Positives (TP), False Positives (FP), False Negatives (FN)를 추출
TP = np.diag(cm)  # 혼동 행렬의 주 대각선에 해당하는 값들이 True Positives
FP = cm.sum(axis=0) - TP  # 각 열의 합에서 True Positives를 뺀 것이 False Positives
FN = cm.sum(axis=1) - TP  # 각 행의 합에서 True Positives를 뺀 것이 False Negatives

# Precision 계산
precision = TP / (TP + FP)

# Recall 계산
recall = TP / (TP + FN)

# Precision 출력
print("Precision for each class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {precision[i]:.2f}")

# Recall 출력
print("Recall for each class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {recall[i]:.2f}")

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def detect_face(image):
    faces, gray = face_rcg.detect_faces(image)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
    return gray[y:y+h, x:x+w], (x, y, w, h)

def predict_face(face, pca, knn_model):
    preprocessor = Preprocessor()
    face_eq, face_recon = preprocessor.preprocess(face)
    face_resized = cv2.resize(face_recon, (IMG_SIZE, IMG_SIZE))
    face_flat = face_resized.flatten()
    face_pca = pca.transform([face_flat])
    predicted_label = knn_model.predict(face_pca)
    return predicted_label[0]

# 웹캠을 엽니다.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    face, bbox = detect_face(frame)
    
    if face is not None:
        predicted_label = predict_face(face, pca, km)
        label_name = [k for k, v in label_dict.items() if v == predicted_label][0]
        (x, y, w, h) = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face Classification', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
