
import cv2
import numpy as np

img_1 = cv2.imread('Un_Buddha_001.bmp')
img_2 = cv2.imread('Un_Buddha_002.bmp')

img_1 = cv2.resize(img_1, (int(img_1.shape[1] / 6), int(img_1.shape[0] / 6)))
img_2 = cv2.resize(img_2, (int(img_2.shape[1] / 6), int(img_2.shape[0] / 6)))

psd_img_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY)
psd_img_2 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)

# 3) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()

psd_kp1, psd_des1 = sift.detectAndCompute(psd_img_1, None)
psd_kp2, psd_des2 = sift.detectAndCompute(psd_img_2, None)

# 4) Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(psd_des1, psd_des2, k=2)
goodMatch = []
for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
print(goodMatch[:20])

img_out = cv2.drawMatchesKnn(img_1, psd_kp1, img_2, psd_kp2, goodMatch[:200], None, flags=2)

cv2.imshow('image', img_out)#展示图片
cv2.waitKey(0)#等待按键按下
cv2.destroyAllWindows()#清除所有窗口
