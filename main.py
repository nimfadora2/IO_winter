import cv2
import numpy as np
from PSO import PSO
from deterministicPSO import deterministicPSO
import time
import sys
import json
import writeResultsToFile as logger

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

config_name = "config.json"
if(len(sys.argv)>1):
	config_name = sys.argv[1]
config_file = open(config_name)
config_data = json.load(config_file)
config_file.close()

image = cv2.imread('lena.png',0)
height, width = image.shape
neigh = neighbors(image)

s = config_data['histogram'][0]['s']
t = config_data['histogram'][0]['t']

deterministic_pso = config_data['testOnDeterministicData']
pso_start_time = time.time()
if(deterministic_pso == False):
	max, s,t = PSO(image, config_data, func=phi)
else:
	max, s,t = deterministicPSO(image, config_data, func=phi)
print('---PSO algorithm duration---', (time.time()-pso_start_time))
print('max, s, t')
print(max,s,t)
for i in range(1,width-1):
	for j in range(1,height-1):
		if image[i,j] > s and neigh[i-1][j-1] > t:
			image[i,j] = 0
		elif image[i,j] < s and neigh[i-1][j-1] < t:
			image[i,j] = 0
		else:
			image[i,j] = 255

score, diff = logger.benchmarkResultImage(image)
logger.writeResultsToFile(image, score, s, t, config_data);

cv2.imshow('I',image)
cv2.waitKey(0)
