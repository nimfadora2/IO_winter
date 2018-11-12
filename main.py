import cv2
import numpy as np
from PSO import PSO
import time

A = cv2.imread('lena.png',0)

### Function to calculate neighbours on image - rowe 0, n-1 are omitted ###
def neighbors(image):
	neighs = np.int_(image[0:-2,0:-2])
	for i in range (0,2,1):
		for j in range(0,2,1):
			if i == 0 and j == 0:
				continue
			current = image[i:i-2,j:j-2]
			neighs = np.add(neighs,current)
	for i in range(0,2):
		current = image[i:i-2,2:]
		neighs = np.add(neighs, current)
		current = image[2:,i:i-2]
		neighs = np.add(neighs, current)
	current = image[2:,2:]
	neighs = np.add(neighs, current)
	neighs = np.int_(np.divide(neighs,9))
	return neighs

### Function to calculate two dimensional histogram based on image and neighs table ###
def hist2D(image, neighs):
	width, height = image.shape
	hist = [[0] * 256] * 256

	for i in range(1, width - 1):
		for j in range(1, height - 1):
			hist[image[i, j]][neighs[i - 1][j - 1]] += 1

	return hist

### Function to calculate probability mass function based on histogram ###
def probab(hist, size):
	return np.divide(hist, size)

### Function to calculate probability distribution to first and second area ###
def probDistr(prob,s,t):
	return np.sum(prob[:s,:t]),np.sum(prob[s:,t:])

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
	hist = hist2D(image,neighs)
	prob = probab(hist, image.size)
	P1, P2 = probDistr(prob, s, t)
	H1, H2 = discrEntr(prob[:3][:4], s, t)
	H_1 = entropy(H1, P1)
	H_2 = entropy(H2, P2)

	return H_1 + H_2


height, width = A.shape
neigh = neighbors(A)

s = 137
t = 138
max = 0
position = (0,0)

first = time.time()
max, s,t = PSO(100,5,A, func=phi)
print('---seconds---',time.time()-first)
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
