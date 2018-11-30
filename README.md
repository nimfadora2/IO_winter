# IO_winter
Image segmentation with use PSO

Short description of included files and implemented functionalities:

config.json -> default configuration file of python application. 'testOnDeterministicData' is a flag specyfing if program should run with random seed (if set to 0) or specific particles position. 

determ.json -> file that can be loaded as parameter into main.py application ('python main.py determ.json'). This file includes also 5 initial positions of particles.

main.py -> program starts from parsing json file (which is obligated, default name is 'config.json' but it also could be passed from terminal). After parsing, two different functions are called depending on 'testOnDeterministicData' field. One function is located in 'PSO.py' file, the other one in 'determiniticPSO.py'. Time of pso algorithm is measured and printed on screen. 
Program ends with comparing results with reference image.

PSO.py -> import random function from numpy and read arguments passed from parsed json configs.

determiniticPSO.py -> full analogy to 'PSO.py'

writeResultsToFile.py -> script that firstly compare results with reference image obtained with Otsu threshold binarization method and then save image and generates file containging information about recent program (in 'result' directory).
