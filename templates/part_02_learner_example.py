import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
from matplotlib import pyplot as plt

# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
# postion 1
C_init = dcm_from_rpy([np.pi/6, -np.pi/8, -np.pi/8])
t_init = np.array([[4.0, -3.0, -0.1]]).T
# postion 2
# C_init = dcm_from_rpy([np.pi/10, -np.pi/10, -np.pi/10])
# t_init = np.array([[1, 1, 0.1]]).T

Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

iter_list = []
gain = np.arange(0.01, 2, 0.01)
for i in gain:
    tmp = ibvs_simulation(Twc_init, Twc_last, pts, K, i, False, False)
    iter_list.append(tmp)

print(f"the minimum number of iterations is {min(iter_list)} corresponding to gain {gain[iter_list.index(min(iter_list))]}")
# 0.97 to 1.15 all have 11 iterations
# 098 to 1.07 all have 11 iterations

plt.figure(figsize=(10, 6))
plt.plot(gain, iter_list, marker='o', linestyle='-', color='r')
plt.title('Figure 5. Number of Iterations versus Gain of Controller with Known Depth for Case 2')
plt.xlabel('Gain Value')
plt.ylabel('Number of Iterations')
plt.grid(True)
plt.show()


# # Camera intrinsics matrix - known.
# K = np.array([[500.0, 0, 400.0], 
#               [0, 500.0, 300.0], 
#               [0,     0,     1]])

# # Target points (in target/object frame).
# pts = np.array([[-0.75,  0.75, -0.75,  0.75],
#                 [-0.50, -0.50,  0.50,  0.50],
#                 [ 0.00,  0.00,  0.00,  0.00]])

# # Camera poses, last and first.
# C_last = np.eye(3)
# t_last = np.array([[ 0.0, 0.0, -4.0]]).T
# C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8])
# t_init = np.array([[-0.2, 0.3, -5.0]]).T

# Twc_last = np.eye(4)
# Twc_last[0:3, :] = np.hstack((C_last, t_last))
# Twc_init = np.eye(4)
# Twc_init[0:3, :] = np.hstack((C_init, t_init))

# gain = 0.1

# # Sanity check the controller output if desired.
# # ...

# # Run simulation - use known depths.
# ibvs_simulation(Twc_init, Twc_last, pts, K, gain, False)