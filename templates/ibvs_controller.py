import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_controller(K, pts_des, pts_obs, zs, gain):
    """
    A simple proportional controller for IBVS.

    Implementation of a simple proportional controller for image-based
    visual servoing. The error is the difference between the desired and
    observed image plane points. Note that the number of points, n, may
    be greater than three. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K       - 3x3 np.array, camera intrinsic calibration matrix.
    pts_des - 2xn np.array, desired (target) image plane points.
    pts_obs - 2xn np.array, observed (current) image plane points.
    zs      - nx0 np.array, points depth values (may be estimated).
    gain    - Controller gain (lambda).

    Returns:
    --------
    v  - 6x1 np.array, desired tx, ty, tz, wx, wy, wz camera velocities.
    """
    v = np.zeros((6, 1))
    

    #--- FILL ME IN ---

    n = pts_des.shape[1]  # Number of desired points

    # Initialize an empty array for stacking the Jacobian matrices
    J = np.empty((0, 6))

    for i in range(n):
        # Compute the Jacobian matrix for each observed point
        
        tmp_J = ibvs_jacobian(K, pts_obs[:, i].reshape(-1, 1), zs[i])
        # Stack the computed Jacobian matrix vertically in J
        # print(zs[i], tmp_J)
        J = np.vstack((J, tmp_J))
    # Compute the pseudo-inverse of the full Jacobian matrix
    J = np.linalg.inv(J.T @ J) @ J.T

    # Calculate the error per point as the difference between desired and observed points
    error_per_point = pts_des - pts_obs
    error_vector = error_per_point.flatten('F')

    # Compute the control velocity using the gain and the product of the pseudo-inverse Jacobian and the error vector
    v = gain * J @ error_vector
    v = v.reshape(-1, 1)  # Reshape for consistency
    #------------------

    correct = isinstance(v, np.ndarray) and \
        v.dtype == np.float64 and v.shape == (6, 1)

    if not correct:
        raise TypeError("In controller, wrong type or size returned!")

    return v