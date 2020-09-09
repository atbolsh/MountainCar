import numpy as np
from copy import deepcopy

class Tiling2D: #Tiling is asymmetric.

    def __init__(self, alpha = 0.1/8, xmin=-1.2, xmax=0.5, vmin=-0.07, vmax=0.07, numTilings=7, numBlocks=8, numActions=3):
        self.alpha = alpha

        self.xmin = xmin
        self.xmax = xmax
        self.vmin = vmin
        self.vmax = vmax
        self.numBlocks = numBlocks
        self.numTilings = numTilings
        self.numActions = numActions
        
        self.xBlockSize = (xmax - xmin)/numBlocks
        self.vBlockSize = (vmax - vmin)/numBlocks
                
        self.weights = []
        for a in range(numActions):
            actionWeights = []
            for tiling in range(self.numTilings):
                tilingWeights = np.zeros([numBlocks+1, numBlocks+1], dtype='float64')
                actionWeights.append(deepcopy(tilingWeights))
            self.weights.append(deepcopy(actionWeights))
    

    def getOffset(self, tilingNum):
        vOffset = self.vBlockSize * ((tilingNum % self.numTilings) / float(self.numTilings))
        xOffset = self.xBlockSize * ((2*tilingNum) % self.numTilings) / float(self.numTilings) #Assymetry here
        
        return xOffset, vOffset


    def getTilingIndex(self, state):
        i = int(state[0]/self.xBlockSize)
        j = int(state[1]/self.vBlockSize)
        return i, j
    

    def getFullIndex(self, state):
        x = max(state[0], self.xmin)
        x = min(x, self.xmax)
        x = x - self.xmin
        
        v = max(state[1], self.vmin)
        v = min(v, self.vmax)
        v = v - self.vmin #This is to rebase the coordinate systems

        fullIndex = []
        for i in range(self.numTilings):
            xOffset, vOffset = self.getOffset(i)
            fullIndex.append(self.getTilingIndex((x + xOffset, v + vOffset)))

        return fullIndex


    def getVal(self, state, actionIndex): #actionIndex is 0, 1, or 2; conversion to action is in the agent, not the dictionary. 
        fi = self.getFullIndex(state)
        
        w = []
        for tiling in range(self.numTilings):
                ind = fi[tiling]
                w.append(self.weights[actionIndex][tiling][ind[0], ind[1]]) # Get the correct block from the tiling
        
        return sum(w)/len(w)
        

    def moveVal(self, state, actionIndex, target):
        current = self.getVal(state, actionIndex)

        fi = self.getFullIndex(state)        
        w = []
        for tiling in range(self.numTilings):
                ind = fi[tiling]
                self.weights[actionIndex][tiling][ind[0], ind[1]] += self.alpha*(target - current)
        
        return None


