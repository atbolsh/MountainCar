import numpy as np
from copy import deepcopy

class MountainEnvironment:
    
    def __init__(self):
        self.x = np.random.random()/5 - 0.6 #Between -0.6 and -0.4
        self.v = 0.0
        self.complete = False
    
    def bound(self):
        if self.x <= -1.2:
            self.x = -1.2
            self.v = 0.0
        if self.x >= 0.5:
            self.x = 0.5
            self.v = 0.0
            self.complete = True
        self.v = max(-0.07, self.v)
        self.v = min(0.07, self.v)
         
    
    def update(self, throttle):
        self.v += 0.001*throttle - 0.0025*np.cos(3*self.x) # See book description
        self.x += self.v
        self.bound()
        if self.complete:
            reward = 0
        else:
            reward = -1
        return self.x, self.v, reward


