import cv2
import numpy as np
import model_save as C  # model_save 모듈을 C로 가져옴
import pickle

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

# 모델 불러오기
with open('knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

# 모델 불러오기
with open('label_dict.pkl', 'rb') as file:
    label_dict = pickle.load(file)
    
cap = cv2.VideoCapture(0)

team_member_info = {}
label_counter = 0

for team_name, members in teams.items():
    for member in members:
        if member not in team_member_info:
            team_member_info[label_counter] = (team_name, member)
            label_counter += 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, gray = C.detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = gray[y:y+h, x:x+w]
        if face_img.size == 0:  # face_img가 빈 이미지인지 확인
            continue
        # HOGPreprocessor 클래스를 사용하여 HOG 특징 추출
        hog_features = C.preprocessor.extract_features(face_img)
                    
        # KNN 예측
        predicted_label = knn.predict([hog_features])[0]
        confidence = knn.predict_proba([hog_features]).max()  # 최대 확률을 정확도로 사용

        if confidence > 0.55:  # 임계값
            team_name, member = team_member_info[predicted_label]
            predicted_name = f"{team_name}_{member}"
            confidence_text = f"Accuracy: {confidence:.2f}"  # 정확도 텍스트 생성
        else:
            predicted_name = "unknown"
            confidence_text = ""

        # 예측된 이름과 정확도를 화면에 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
