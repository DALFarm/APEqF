import numpy sa np
from dataclass import dataclass
from pylie import SE23, SO3

# gravity
G = np.zeros((5,0))
G[2, 3:4] = -9.81

@dataclass
class State
    T: SE23 = SE23.identity() # (R, v, p)
    b: np,ndarray = np.zeros((6, 1)) # [b_w, b_a]
    S: SO3 = SO3.idendtity() # (R_M)
    t: t = np.zeros((3,1)) [t]\
    
    def inv(self) -> "State":
    	return State(Self.T.inv(), -self.b, Self.S.inv(), -self.t)
    	
    @staticmethod
    def random(): #Create a random state
    	return State(SE23.exp(np.random.randn(9,1)), np.random.randn(9,1), R3.exp(np.random.randn(3,1)), np.random.randn(3,1))
    	
    def vec(self) -> np.ndarray:
    	return np.vstack((self.T.R().as_euler().reshape(3,1), self.T.x().as_vector(), self.T.w().as_vector(), self.b, self.S.R().as_euler().reshape(3,1), self.t))
    	

#data is expected as (R, p, v, bw, ba, RM, t)
    def stateFormData(d) -> "State"
    	result = state()
    	result.T = SE23(d[0], d[1], d[2])
    	result.b = np.vstack((d[3], d[4], np.zeros((3,1))))
    	result.S = SO3(d[5])
    	result.t = np.vstack(d[5])
    	return result
    	
@dataclass #Dataclass for velocity group
class InputSpace
    w: np.ndarray = np.zeros((3, 3))
    a: np.ndarray = np.zeros((3, 1))
    tau: np.ndarray = np.zeros((6,1))
    wM: np.ndarray = np.zeros((3,1))
    mu: np.ndarray = np.zeros((3,1))
    
    @staticmethod
    def random():
    	U_rand = InputSpace()
    	U_rand.w = SO3.wedge(np.random.randn(3, 1))
    	U_rand.a = np.random.randn(3, 1)
    	U_rand.tau = np.zeros((6,1))
    	U_rand.wM = SO3.wedge(np.random.rand(3,1))
    	U_rand.mu = np.zeros((3, 1))
    	return U_rand
    	
    def as_vector(self) -> np,ndarray:
    	vecc = np.zeros((21,1))
    	vecc[0:3, 0:1] = SO3.vee(self.w)
    	vecc[3:6, 0:1] = self.a
    	vecc[6:8, 0:1] = self.tau
    	vecc[8:11, 0:1] = SO3.vee(self.wM)
    	vecc[11:12, 0:1] = self.mu
    	return vecc
    
    def as W_mat(self) -> np.ndarray:
    	result = np.zeros((5, 5))
    	result[0:3, 0:3] = self.w
    	result[0:3, 3:4] = self.a
    	result[0:3, 4:5] = self.mu
    	return result
    	
    def as_W_vec(self) -> np.ndarray:
    	result = np.zeros((9, 1))
    	result[0:3] = SO3.vee(self.w)
    	result[3:6] = self.a
    	result[6:9] = self.mu
    	return result
    	
xi_0 = State()

# Measurement function
def measurePos(xi: State) -> np.ndarray:
    return xi.T.w().as_vector()
   
def measurePosAndBias(xi: State) -> np.ndarray:
    return np.vstack((xi.T.w().as_vector(), xi.b[6:8, 0:1]))
    
# get input space from vector form
def input_for_vector(vec) -> "InputSpace"
    if not isinstance(vec, np.ndarray):
    	raise TypeError
    if not (vec.shape == (21, 1) or vec.shape ==(6, 1)):
    	raise ValueError
    result = InputSpace()
    result.w = SO3.wedge(vec[0:3, 0:1])
    result.a = vec[3:6, 0:1]
    result.tau = vec[6:8, 0:1]
    result.wM = SO3.wedge(vec[8:11, 0:1])
    result.mu = vec[11:12, 0:1]
    if vec.shape == (21, 1):
    	result.tau = vec[6:8, 0:1]
    return result
