import numpy as np
from numpy.linalg import matrix_rank
from ibvs_jacobian import *

# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])
pt = np.array([[200.0], [400.0]])
z  = 1.0

# Jacobian should have rank 2.
J = ibvs_jacobian(K, pt, z)
print("Rank: " + str(matrix_rank(J)))