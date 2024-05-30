import numpy as np
import cv2
import os


# 사용할 특잇값의 수
k = 100

# 얼굴 감지기 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 팀원 이름과 번호 리스트
team_1_members = ['001_PIW', '002_KHH', '003_CSB', '004_KDM', '005_LSE']
# team_2_members = ['001_AAA', '002_BBB', '003_CCC', '004_DDD']

# 팀 목록
teams = {
    'Team1': team_1_members,
    # 'Team2': team_2_members
}

dataset_path = 'dataset'  # 데이터셋 디렉토리 경로

# 데이터를 저장할 리스트 초기화
X_list = []
y_list = []

label_dict = {member: idx for idx, member in enumerate(team_1_members)}

for team_name, members in teams.items():
    team_path = os.path.join(dataset_path, team_name)  # 각 팀의 디렉토리 경로
    for member in members:
        member_id = member.split('_')[0]
        member_name = member.split('_')[1]
        member_path = os.path.join(team_path, member)  # 각 멤버의 디렉토리 경로
        for photo_num in range(1, 99):  # 사진의 개수는 조정
            image_filename = f'{team_name}_{member_id}_{member_name}_{photo_num:04d}.jpg'  # 이미지 파일 이름
            image_path = os.path.join(member_path, image_filename)  # 이미지 파일 경로

            # 경로 출력 및 존재 여부 확인
            print(f"Checking: {image_path}")
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {image_path} not found.")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 얼굴 감지
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # 출력을 위한 이름에서 숫자와 언더바 제거
            member_display_name = member_name

            # 각 얼굴에 대해 특이값 분해 수행
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]  # 얼굴 부분 이미지 추출
                U, S, VT = np.linalg.svd(face_img.astype(float))  # 특이값 분해 수행

                # 상위 k개의 특잇값을 포함하는 고정된 길이의 벡터로 변환
                feature_vector = np.concatenate((S[:k], np.zeros(max(0, k - len(S)))), axis=0)

                # 데이터 리스트에 추가
                X_list.append(feature_vector)
                y_list.append(label_dict[member])




