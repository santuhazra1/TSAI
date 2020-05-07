import numpy as np
from random import random, randint, choice

class RandomLocation(object):

    def __init__(self):

        # Define multiple Target in the dictionary. Total 12 Target defined
        self.TargetDict = {'A': [168, 111], 'B': [117, 445], 'C': [361, 308], 'D': [610, 30], 'E': [582, 532], 'F': [498, 413], 'G': [669, 214], 'H': [811, 479],\
            'I': [878, 623], 'J': [1015, 137], 'K': [1092, 367], 'L': [1395, 424]}

        # Define multiple Location in the dictionary. Total 12 Location defined    
        self.CarLocationDict =  {'A': [138, 276], 'B': [244, 555], 'C': [395, 87], 'D': [338, 480], 'E': [581, 347], 'F': [744, 49], 'G': [808, 239], 'H': [888, 385],\
            'I': [661, 582], 'J': [1174, 267], 'K': [1045, 595], 'L': [1335, 536]}	

        self.LocationName = ['A','B','C','D','E','F','G','H','I','J','K','L']

    def target(self):
        # select a random location for target
        TargetName = choice(self.LocationName)
        return self.TargetDict[TargetName]

    def carlocation(self):
        # select a random location for Car Location
        LocationName = choice(self.LocationName)
        return self.CarLocationDict[LocationName]        


