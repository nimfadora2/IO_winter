import cv2
import numpy as np
from PSO import PSO

A = cv2.imread('lena.png',0)

### Function to calculate neighbours on image - rowe 0, n-1 are omitted ###
def neighbors(image):
	width, height = image.shape
	neighs = []
	for i in range(1, width - 1):
		current_row = []
		for j in range(1, height - 1):
			current_neigh = np.sum(image[i-1:i+2,j-1:j+2])//9
			current_row.append(current_neigh)
		neighs.append(current_row)
	return neighs

### Function to calculate two dimensional histogram based on image and neighs table ###
def hist2D(image,neighs):
	width, height = image.shape
	hist = [[0]*256]*256

	for i in range(1, width - 1):
		for j in range(1, height - 1):
			hist[image[i,j]][neighs[i-1][j-1]]+= 1

	return hist

### Function to calculate probability mass function basen on histogram ###
def probab(hist, size):
	prob = []
	for row in hist:
		prob.append([x/size for x in row])
	return prob

### Function to calculate probability distribution to first and second area ###
def probDistr(prob,s,t):
	return np.sum(prob[:s][:t]),np.sum(prob[s:][t:])

### Function to calculate dicrete entropy of both areas ###
def discrEntr(prob,s,t):
	H1 = 0
	H2 = 0
	for row in prob[:s]:
		for elem in row[:t]:
			if elem != 0:
				H1 += elem*np.log(elem)
	for row in prob[s:]:
		for elem in row[t:]:
			if elem != 0:
				H2 += elem*np.log(elem)
	return -H1,-H2

### Function to calculate entropy of area ###
def entropy(H,P):
	if P == 0:
		return 0
	return np.log(P) + H/P

### Function to calculate phi on given s and t ###
def phi(image,s,t):
	neighs = neighbors(image)
	hist = hist2D(image, neighs)
	prob = probab(hist, image.size)
	P1, P2 = probDistr(prob, s, t)
	H1, H2 = discrEntr(prob, s, t)
	H_1 = entropy(H1, P1)
	H_2 = entropy(H2, P2)

	return H_1 + H_2

height, width = A.shape
neigh = neighbors(A)

s = 137
t = 138
max = 0
position = (0,0)
max, s,t = PSO(70,5,A, func=phi)
print(max,s,t)
for i in range(1,width-1):
	for j in range(1,height-1):
		if A[i,j] > s and neigh[i-1][j-1] > t:
			A[i,j] = 0
		elif A[i,j] < s and neigh[i-1][j-1] < t:
			A[i,j] = 0
		else:
			A[i,j] = 255

cv2.imshow('I',A)
cv2.waitKey(0)