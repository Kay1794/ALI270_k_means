import numpy as np
import cv2

img = cv2.imread('./BSR/BSDS500/data/images/train/113016.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
label1 = label.reshape((img.shape[0],img.shape[1]))
contour = np.zeros((img.shape))
print(res2.shape)
print(contour.shape)
for i in range(img.shape[0]-1):
	for j in range(img.shape[1]-1):
		if (label1[i][j]!= label1[i][j+1]) or (label1[i][j]!= label1[i+1][j]):
			res2[i][j] = [0,0,0]
		else:
			contour[i][j] = res2[i][j]

			#print(contour[i][j])
#print(contour)
#contour = contour.astype(np.uint8)
#np.savetxt("/Users/zhongyiqi/Documents/ALI270/Midterm_Planning_Form/slides_code/contour.txt",contour)

cv2.imshow('contour',contour)
cv2.waitKey(0)
cv2.imshow('res2',res2)
cv2.imwrite("/Users/zhongyiqi/Documents/ALI270/Midterm_Planning_Form/slides_code/k_means_horse"+".jpg",res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
