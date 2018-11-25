import numpy as np
import sys
import json
from skimage.measure import compare_ssim
import cv2

### Function to compare pso algorithm results with autothresholded lists of images. ###
def benchmarkResultImage(image):
	print('Result benchmark phase')
	ref_img = cv2.imread('reference.png', 0)
	(score, diff) = compare_ssim(ref_img, image, full=True)
	diff = (diff * 255).astype("uint8")
	print("SSIM: {}".format(score))
	return(score, diff)

### Function to save Numpy Int 64 to json like file ###
def saveNumpy(o):
    if isinstance(o, np.int64): return int(o)

### Function to write results and image to file ####
def writeResultsToFile(image, score, s, t, config_data):	
	if not os.path.exists('./pso_results'):
    		os.makedirs('./pso_results')
	result_name = './pso_results/result'
	result_name += str(abs(score))[2:7]
	result_name += '_' + str(s) + '_' + str(t)
	with open(result_name+'.json', 'w') as fp:
		json.dump({"benchmark_results":[{'ssim_score': score}], "histogram_result":[{'s':s}, {'t':t}], "psoInit": config_data['psoInit'][0]}, fp, indent=4, default=saveNumpy)
	cv2.imwrite(result_name + '.png', image)
