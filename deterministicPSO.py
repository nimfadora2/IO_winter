import random
import numpy as np
import sys
import json

random.seed()

### Initialization of positions ###
def readInitPositionsFromFile(image, m, config_data):
    width, height = image.shape
    particles = []
    for i in range(m):
        x = config_data['particlesPosition'][i]['x']
        y = config_data['particlesPosition'][i]['y']
        if((x in range(0,width)) and (y in range (0,height))):
            particles.append([x,y])
        else:
            sys.exit("Check config file. Init particle position out of range (0,sizeofImage). Program ends. Image [height,width]: ", height, width)
    return particles

### Initialization of velocity vector ###
def initVel(m, V_max):
	return [(random.uniform(0,V_max),random.uniform(0,V_max)) for i in range(m)]

def deterministicPSO(image, config_data, func):
	pso_params = config_data['psoInit'][0]
	number_of_partitions = pso_params['numberOfPartitions']
	velocity_max = pso_params['velocityMax']
	positions = readInitPositionsFromFile(image,number_of_partitions, config_data)
	best = [(func(image,elem[0],elem[1]),elem[0],elem[1]) for elem in positions]
	global_best = max(best, key=lambda x:x[0])
	vel = initVel(number_of_partitions, velocity_max)
	W = pso_params['w']
	c1 = pso_params['c1']
	c2 = pso_params['c2']
	for i in range(pso_params['iterations']):
		vel = [
			np.multiply(W,vel[i])+
			np.multiply(np.multiply(c1,[random.uniform(0,1), random.uniform(0,1)]),
			(np.subtract(best[i][1:2],positions[i])))+
		    np.multiply(np.multiply(c2,[random.uniform(0,1), random.uniform(0,1)]),
			(np.subtract(global_best[1:2], positions[i])))
			for i in range(number_of_partitions)]
		positions = [np.int_(np.add(positions[i],vel[i])) for i in range(number_of_partitions)]
		best = [(func(image,elem[0], elem[1]), elem[0], elem[1]) for elem in positions]
		global_best = max(best+[global_best], key=lambda x: x[0])
		vel = initVel(number_of_partitions, velocity_max)
	return global_best