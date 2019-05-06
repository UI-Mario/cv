import cv2
import math
import numpy as np

def MoravecCorners(I, kSize, threshold, i_np, j_np):
	M = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	H = np.array(M)
	rows = I.shape[0]
	cols = I.shape[1]
	# print(img.shape)
	r = int(kSize/2)
	# 兴趣值
	n = 0

	for i in range(r, rows-r):
		for j in range(r, cols-r):
			v = [0, 0, 0, 0]
			for k in range(-r,r):
				v[0] += pow(int(H[i, j+k]) - int(H[i, j+k+1]), 2)
				v[1] += pow(int(H[i+k, j]) - int(H[i+k+1, j]), 2)
				v[2] += pow(int(H[i+k, j+k]) - int(H[i+k+1, j+k+1]), 2)
				v[3] += pow(int(H[i+k, j-k]) - int(H[i+k+1, j-k-1]), 2)
			value = min(v)
			if value > threshold:
				n+=1
				#圆的绘制
				#第一个参数  图片的数据
				#第二个参数  圆心
				#第三个参数  半径
				#第四个参数  颜色
				#第五个参数  填充或线宽
				i_np[n]=j
				j_np[n]=i
				#cv2.circle(H,(i,j),3,(211,0,213),1)
	for m in range(0,n):
		cv2.circle(H,(i_np[m],j_np[m]),3,(211,0,213),1)
	return H,i_np,j_np

img = cv2.imread('1.jpg', 1)

i_np = [0]*100000
j_np = [0]*100000
i_1=i_np
j_1=j_np

res, i_1, j_1 = MoravecCorners(img, 5, 10000, i_np, j_np)
cv2.imshow('temp', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
