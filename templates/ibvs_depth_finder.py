import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian
import scipy

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    # J = np.zeros((2*n, 6))
    zs_est = np.ones((n,))

    #--- FILL ME IN ---

    # Extract intrinsic parameters
    fx = K[0, 0]  # focal length (x-axis)
    fy = K[1, 1]  # focal length (y-axis)
    cx = K[0, 2]  # principal point (x-coordinate)
    cy = K[1, 2]  # principal point (y-coordinate)

    # Extract velocities
    v = v_cam[0:3, :]  # Linear velocity of the camera
    w = v_cam[3:6, :]  # Angular velocity of the camera

    for i in range(n):
        # Compute the Jacobian matrix for each point
        J = ibvs_jacobian(K, pts_obs[:, i].reshape(-1, 1), zs_est[i])

        # Calculate the displacement of each point from previous to current frame
        displacement = pts_obs[:, i] - pts_prev[:, i]

        # Compute the effects of linear and angular velocities on point displacement
        A = J[:, 0:3] @ v  # Effect of linear velocity
        B = displacement.reshape(-1, 1) - J[:, 3:6] @ w  # Displacement minus effect of angular velocity

        # Solve for depth using least squares approach
        depths = scipy.linalg.lstsq(A, B)[0]
        zs_est[i] = 1 / depths  # Invert to get the estimated depth

    #------------------

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est