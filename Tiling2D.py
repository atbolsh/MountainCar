import numpy as np
from copy import deepcopy

class Tiling2D: #Tiling is asymmetric, 4 in x, 2 in v

    def __init__(self, alpha = 0.1/8, xmin=-1.2, xmax=0.5, vmin=-0.07, vmax=0.07, numBlocks = 8, numActions=3):
        self.alpha = alpha

        self.xmin = xmin
        self.xmax = xmax
        self.vmin = vmin
        self.vmax = vmax
        self.numBlocks = numBlocks
        self.numTilings = 8
        self.numActions = numActions
        
        self.xBlockSize = (xmax - xmin)/numBlocks
        self.vBlockSize = (vmax - vmin)/numBlocks
        
        self.xNumOffsets = 4
        self.vNumOffsets = 2
        
        self.xOffset = self.xBlockSize/self.xNumOffsets
        self.vOffset = self.vBlockSize/self.vNumOffsets
        
        self.weights = []
        for a in range(numActions):
            actionWeights = []
            for xTiling in range(self.xNumOffsets):
                xTilingWeights = []
                for vTiling in range(self.vNumOffsets):
                    vTilingWeights = np.zeros([numBlocks+1, numBlocks+1], dtype='float64')
                    xTilingWeights.append(deepcopy(vTilingWeights))
                actionWeights.append(deepcopy(xTilingWeights))
            self.weights.append(deepcopy(actionWeights))
                    

    def getTilingIndex(self, x, v):
        i = int(x/self.xBlockSize)
        j = int(v/self.vBlockSize)
        return i, j
    

    def getFullIndex(self, x, v):
        x = max(x, self.xmin)
        x = min(x, self.xmax)
        x = x - self.xmin
        
        v = max(v, self.vmin)
        v = min(v, self.vmax)
        v = v - self.vmin #This is to rebase the coordinate systems

        fullIndex = []
        for xTiling in range(self.xNumOffsets):
            xInds = []
            for vTiling in range(self.vNumOffsets):
                vInds = self.getTilingIndex(x + self.xOffset*xTiling, v + self.vOffset*vTiling)
                xInds.append(deepcopy(vInds))
            fullIndex.append(deepcopy(xInds))
        return fullIndex


    def getVal(self, x, v, actionIndex): #actionIndex is 0, 1, or 2; conversion to action is in the agent, not the dictionary. 
        fi = self.getFullIndex(x, v)
        
        w = []
        for xTiling in range(self.xNumOffsets):
            for vTiling in range(self.vNumOffsets):
                ind = fi[xTiling][vTiling]
                w.append(self.weights[actionIndex][xTiling][vTiling][ind[0], ind[1]]) # Get the correct block from the tiling
        
        return sum(w)/len(w)
        

    def moveVal(self, x, v, actionIndex, target):
        fi = self.getFullIndex(x, v)
       
        for xTiling in range(self.xNumOffsets):
            for vTiling in range(self.vNumOffsets):
                ind = fi[xTiling][vTiling]
                self.weights[actionIndex][xTiling][vTiling][ind[0], ind[1]] += self.alpha*(target - self.weights[actionIndex][xTiling][vTiling][ind[0], ind[1]])
        
        return None
