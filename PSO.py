import random
import numpy as np

random.seed()

### Initialization of positions ###
def initPos(image, m):
	width, height = image.shape
	particles = []
	for i in range(m):
		(x,y) = (random.randint(0,width-1),random.randint(0,height-1))
		while (x,y) in particles:
			(x, y) = (random.randint(0, width-1), random.randint(0, height-1))
		particles.append([x,y])
	return particles

### Initialization of velocity vector ###
def initVel(m, V_max):
	return [(random.uniform(0,V_max),random.uniform(0,V_max)) for i in range(m)]

def PSO(iterations, m, image, func, W=0.8, V_max = 2.5, c1=1.3, c2=1.3):
	positions = initPos(image,m)
	best = [(func(image,elem[0],elem[1]),elem[0],elem[1]) for elem in positions]
	global_best = max(best, key=lambda x:x[0])
	vel = initVel(m, V_max)
	for i in range(100):
		vel = [
			np.multiply(W,vel[i])+
			np.multiply(np.multiply(c1,[random.uniform(0,1), random.uniform(0,1)]),
			(np.subtract(best[i][1:2],positions[i])))+
		    np.multiply(np.multiply(c2,[random.uniform(0,1), random.uniform(0,1)]),
			(np.subtract(global_best[1:2], positions[i])))
			for i in range(m)]
		positions = [np.int_(np.add(positions[i],vel[i])) for i in range(m)]
		best = [(func(image,elem[0], elem[1]), elem[0], elem[1]) for elem in positions]
		global_best = max(best+[global_best], key=lambda x: x[0])
		vel = initVel(m, V_max)
	return global_best