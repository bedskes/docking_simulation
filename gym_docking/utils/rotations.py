import numpy as np

def rotation2(angle):
    '''
    angle: Angle of 2D rotation in radians

    Returns a 2x2 numpy array representing the 2D rotation matrix for the given angle.
    '''
    mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return mat

def rotation3(axis, angle):
    '''
    axis: axis of rotation (0, 1, or 2).
    angle: Angle of 2D rotation in radians

    Returns a 3x3 numpy array representing the 3D rotation matrix for the given
    angle about the specified axis.
    '''
    mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    mat = np.concatenate((mat[:axis], np.zeros((1,2)), mat[axis:]), axis = 0)
    mat = np.concatenate((mat[:,:axis], np.zeros((3,1)), mat[:,axis:]), axis = 1)
    mat[axis, axis] = 1.0
    return mat

def rotation_all(angles):
    '''
    angles: list or tuple of length 3, one value per dimension, representing the
            angles of rotation about the corresponding axis in radians.

    Returns a 3x3 numpy array representing the 3D rotation matrix for the given
    angles about their specified axes. Rotations are performed in order by index.
    '''
    mat = np.identity(3)
    for i in range(0, 3):
        mat = np.matmul(mat, rotation3(i, angles[i]))
    return mat

def rotation_all_rev(angles):
    '''
    angles: list or tuple of length 3, one value per dimension, representing the
            angles of rotation about the corresponding axis in radians.

    Returns a 3x3 numpy array representing the 3D rotation matrix for the given
    angles about their specified axes. Rotations are performed in reverse order
    by index.
    '''
    mat = np.identity(3)
    for i in range(2, -1, -1):
        mat = np.matmul(mat, rotation3(i, angles[i]))
    return mat