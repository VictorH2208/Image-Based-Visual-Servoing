import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
import matplotlib.pyplot as plt
from numpy.linalg import inv

def plot_image_points(pts_des, pts_obs):
    """Plot observed and desired image plane points."""
    plt.clf()
    print(pts_obs[0:1, :], pts_obs[1:2, :])
    plt.plot(pts_des[0:1, :], pts_des[1:2, :], 'rx')
    plt.plot(pts_obs[0:1, :], pts_obs[1:2, :], 'bo')
    plt.xlim([-3500, 600])
    plt.ylim([0, 3000])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.title('Figure 4. Initial and Desired Points in Image Plane in Case 2')
    plt.show()
    # plt.pause(0.2)

# Project points into camera. Returns truth depths.
def project_into_camera(Twc, K, pts):
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))
    pts_cam = (inv(Twc)@pts)[0:3, :]
    zs = pts_cam[2, :] 
    pts_cam = K@pts_cam/pts_cam[2:3, :]
    return pts_cam[0:2, :], zs

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
# position 1
C_init = dcm_from_rpy([np.pi/6, -np.pi/8, -np.pi/8])
t_init = np.array([[4.0, -3.0, -0.1]]).T
# postion 2
# C_init = dcm_from_rpy([np.pi/10, -np.pi/10, -np.pi/10])
# t_init = np.array([[1, 1, 0.1]]).T


Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))
# print(Twc_init)
# print(Twc_last)

pts_des, _ = project_into_camera(Twc_last, K, pts)
pts_obs, zs = project_into_camera(Twc_init, K, pts)

plot_image_points(pts_des, pts_obs)

# iter_list = []
# gain = np.arange(0.01, 1.4, 0.01)
# for i in gain:
#     tmp = ibvs_simulation(Twc_init, Twc_last, pts, K, i, True, False)
#     iter_list.append(tmp)

# print(f"the minimum number of iterations is {min(iter_list)} corresponding to gain {gain[iter_list.index(min(iter_list))]}")
# # => 0.97 to 1.04 10 iter for postion 1
# # => 0.96 to 1.19 12 iter for postion 2

# plt.figure(figsize=(10, 6))
# plt.plot(gain, iter_list, marker='o', linestyle='-', color='r')
# plt.title('Number of Iterations vs Gain')
# plt.xlabel('Gain Value')
# plt.ylabel('Number of Iterations')
# plt.grid(True)
# plt.title('Figure 6. Number of Iterations versus Gain of Controller with Estimated Depth for Case 2')
# plt.show()

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

# # Run simulation - estimate depths.
# ibvs_simulation(Twc_init, Twc_last, pts, K, gain, True)

