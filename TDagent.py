from Tiling2D import *
from MountainEnvironment import *

class Agent:

    def __init__(self, initial = (-0.5, 0), gamma = 1.0, alpha = 0.1/8, eps=0.1):
        self.actions = [-1, 0, 1]

        self.initial = initial
        self.reset()
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.Q = Tiling2D(alpha=alpha, numActions = len(self.actions))

    def reset(self, env=None):
        if type(env) == type(None):
            self.current = self.initial
        else:
            self.current = (env.x, env.v)
        self.nextAction = None

    def inclusiveArgMax(self, l):
        M = -1000000
        inds = []
        for i in range(len(l)):
            v = l[i]
            if v > M:
                M = v
                inds = [i]
            elif v == M:
                inds.append(i)
        return inds

    def greedyAction(self, state=None):
        if type(state) == type(None):
            state = self.current
        vals = [self.Q.getVal(state, ind) for ind in range(len(self.actions))] # In this case, all actions available always; change back if needed.
        inds = self.inclusiveArgMax(vals)
        ind = np.random.choice(inds)
        return self.actions[ind]

    def exploringAction(self): #No need to see env or state info; choose random action.
        return np.random.choice(self.actions)

    def expectation(self, state=None):
        if type(state) == type(None):
            state = self.current
        v = [self.Q.getVal(state, ind) for ind in range(len(self.actions))]
        re = self.eps*sum(v)/len(v)
        rg = (1 - self.eps)*max(v)
        return re + rg
 
    def action(self, state=None):
        if type(state) == type(None):
            state = self.current

        if np.random.random() < self.eps:
            a = self.exploringAction()
        else:
            a = self.greedyAction(state)
        return a

    def indexFromAction(self, action): #Hack to get agreement; find general method please
        return action + 1

    def move(self, env):
        
        if type(self.nextAction) == type(None):
            self.nextAction = self.action()
        a = self.nextAction
        ind1 = self.indexFromAction(a)
        
        newState, R = env.move(self.current, a) 

        self.nextAction = self.action(newState)
        ind2 = self.indexFromAction(self.nextAction)

        newQ = self.Q.getVal(newState, ind2)

        target = R + (self.gamma * newQ)
        self.Q.moveVal(self.current, ind1, target)

        self.current = newState
       
        return a, R


        
 

