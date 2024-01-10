import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---
    # Extract intrinsic parameters
    fx = K[0, 0]  # focal length
    fy = K[1, 1]  # focal length

    # Normalize image coordinates (u, v)
    homo_pt = np.vstack((pt, np.ones((1, pt.shape[1]))))
    normalized_pt = np.linalg.inv(K) @ homo_pt
    normalized_pt = normalized_pt[:2, :] / normalized_pt[2]

    # Calculate the u and c based on equations
    u = normalized_pt[0, 0] * fx
    v = normalized_pt[1, 0] * fy

    # Compute the Jacobian matrix using the provided equations
    J = np.array([
        [-(fx/z), 0, u/z, u*v/fx, -(fx + u**2/fx), v],
        [0, -(fx/z), v/z, fx + v**2/fx, -u*v/fx, -u]
    ], dtype=np.float64)
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("In jacobian, wrong type or size returned!")
    
    return J