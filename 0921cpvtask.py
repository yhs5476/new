import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 읽기 (OpenCV 사용)
img = cv2.imread("soccer.jpg")

# 평평하게 만들기 (3차원 배열을 1차원 배열로 변환)
flat_img = img.flatten()

# 히스토그램 계산 (픽셀 값의 분포 확인)
hist, bins = np.histogram(flat_img, 256, [0, 256])

# 누적 분포 함수 (CDF) 계산
# - hist 배열의 각 원소는 해당 픽셀 값의 빈도 수를 나타냄
# - cumsum 함수를 사용하여 연속적인 누적 합을 계산
cdf = hist.cumsum()

# CDF 정규화
# - cdf 값을 최대값으로 나누어 0~1 사이의 범위로 조정
cdf_normalized = cdf * hist.max() / cdf.max()

# 히스토그램과 누적 분포 함수 (CDF) 시각화 (Matplotlib 사용)
plt.plot(cdf_normalized, color='b', label='CDF')  # 파란색 선으로 CDF 표시
plt.hist(flat_img, 256, [0, 256], color='r', label='hist')  # 빨간색 막대로 히스토그램 표시
plt.xlim([0, 256])  # x축 범위 설정 (0~256)
plt.legend(('CDF', 'hist'), loc='upper left')  # 범례 설명 추가 (왼쪽 상단)
plt.show()  # 플롯 출력

# 히스토그램 평활화를 위한 누적 분포 함수 (CDF) 변환
# - 0 값을 가진 원소를 마스킹 (무시)
cdf_m = np.ma.masked_equal(cdf, 0)

# 최소값에서 빼고 최대값 범위에 맞게 스케일링
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

# 마스킹 해제 후 uint8 타입으로 변환
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

# 히스토그램 평활화 적용
img2 = cdf[img]  # 원본 이미지에 변환된 CDF 값을 매핑

# 평활화된 이미지의 히스토그램 계산
hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
cdf = hist.cumsum()

# 정규화된 CDF 계산
cdf_normalized = cdf * hist.max() / cdf.max()

# 평활화된 이미지의 히스토그램과 CDF 시각화
plt.plot(cdf_normalized, color='b', label='CDF')
plt.hist(img2.flatten(), 256, [0, 256], color='r', label='hist')
plt.xlim([0, 256])
plt.legend(('CDF', 'hist'), loc='upper left')
plt.show()

# 원본 이미지와 히스토그램 평활화된 이미지 출력 (OpenCV 사용)
cv2.imshow("Original Image", img)
cv2.imshow("Histogram Equalized Image", img2)
cv2.waitKey(0)  # 창을 닫을 때까지 키 입력 대기

# 히스토그램 평활화된 이미지 저장
cv2.imwrite("img2", img2)
cv2.waitKey(0)  # 창을 닫을 때까지 키 입력 대기