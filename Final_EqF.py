from scipy.linalg import expm
from Symmetry import *
from pylie import SE23

def continuous_lift(xi : State, U : InputSpace) -> np.ndarray:
    L = np.zeros((21, 1))
    L[]
