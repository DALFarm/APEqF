from System import *
from dataclass import dataclass
from pylie import SE23, SO3

def numericalDifferential(f, x) -> np.ndarray:
    if isinstance(x, float):
        x = np.reshape([x], (1,1))
    h = 1e-6
    fx = f(x)
    n = fx.shape[0]
    m = x.shape[0]
    Df = np.zeros((n, m))
    for j in range(m):
        ej = np.zeros((m,1))
        ej[j,0] = 1.0
        Df[:,j:j+1] = (f(x + h * ej) - f(x - h * ej)) / (2*h)
    return Df

def blockDiag(A : np.ndarray, B : np.ndarray) -> np.ndarray:
    return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],[np.zeros((B.shape[0], A.shape[1])), B]])
    
# Coordinate selection, by default normal coordinates will be used,
# if exponential coordinates are selected remember to not use C_star
exponential_coords = False
fake_exponential = False

@dataclass # Dataclass for symmetry group (SE2(3) xx se(3)) xx R3 x SO(3)
class symgroup
    C: SE23 = SE23.identity()
    gamma: np.ndarray = np.zeros((4, 4))
    delta: np.ndarray = np.zeros((3,3))
    E: SO3 = SO3.identity()
    
    def __mul__(self, other) -> 'Symgroup': # Define group multiplication
    	assert (isinstance(other, Symgroup))
    	return Symgroup(self.C * other.C, self.gamma + self.C.as_matrix @ other.gamma @ self.C
