import numpy as np
import FeatureExtraction as FE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib

# 배열로 변환
X = np.array(FE.X_list)
y = np.array(FE.y_list)

# 학습용과 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 생성
knn = KNeighborsClassifier(n_neighbors=5)  # 이웃의 개수를 지정.

# 모델 학습
knn.fit(X_train, y_train)

# 모델 평가
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)


# # 정확도를 그래프로 시각화
# plt.bar(['Accuracy'], [accuracy], color=['blue'])
# plt.ylim(0, 1)
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy')
# plt.show()



# 테스트 데이터에 대한 예측
y_pred = knn.predict(X_test)


# 모델 저장
joblib.dump(knn, './knn_model.pkl')

# 혼동 행렬 생성
cm = confusion_matrix(y_test, y_pred)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=FE.team_1_members, yticklabels=FE.team_1_members)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
