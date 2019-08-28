import os
import numpy as np
import cv2

def combine_R_t(R,t):
    return np.concatenate((R,np.array([t]).T),axis=1)

def R_t_from_P(P):
    R = P[:,0:3]
    t = P[:,3]
    return R,t

def transform_invariance_normalisation(x1, x2):
    """
    This function scales and translates the points so that the centroid of the reference points
    is at the origin of the coordinates and the RMS distance of the points from the origin is equal to √2.
    Note: This is not for normalisation using the calibration matrix (i.e. x_norm = K_inv @ x).
    """
    assert x1.shape[1] == 2
    assert x2.shape[1] == 2
    assert x1.shape[0] == x2.shape[0]
    s = x1.shape[0]

    tran = -np.mean(x1,axis=0)
    scal = np.sqrt(2) / np.mean( sqrt( sum((x1 + tran)**2,axis=1)) )

    T1 = [  [scal,   0,      scal*tran[0]],
            [0,      scal,   scal*tran[1]],
            [0,      0,      1]]

    tran = -np.mean(x2,axis=0)
    scal = np.sqrt(2) / np.mean( sqrt( sum((x2 + tran)**2,axis=1)) )

    T2 = [  [scal,   0,      scal*tran[0]],
            [0,      scal,   scal*tran[1]],
            [0,      0,      1]]

    x1_homo = np.column_stack([x1, np.ones(src.shape[0])])
    x2_homo = np.column_stack([x2, np.ones(src.shape[0])])

    x1_t = (T1 @ x1_homo.T).T;
    x2_t = (T2 @ x2_homo.T).T;

    return x1_t, x2_t, T1, T2

def traingulate_point(P1,P2,x1,x2):
    """NOTE: Think incorrect? using cv2.triangulatePoints instead"""
    A = np.array([(x1[0] * P1[2,:]) - P1[0,:],
                  (x1[1] * P1[2,:]) - P1[1,:],
                  (x2[0] * P2[2,:]) - P2[0,:],
                  (x2[1] * P2[2,:]) - P2[1,:]])

    _,_,v = np.linalg.svd(A)
    X = v[:,-1]
    X_3d = X[0:3] / X[-1]
    return X_3d

def triangulate_points(P1,P2,x1s_unnorm,x2s_unnorm):
    """NOTE: Think traingulate_point is incorrect? using cv2.triangulatePoints instead
    also, need to unnormalise points?"""
    x1s_norm, x2s_norm, T1, T2 = transform_invariance_normalisation(x1s_unnorm, x2s_unnorm)
    P1 = T1 @ P1
    P2 = T2 @ P2
    x1s = [x1s_norm[:,0:2]]
    x2s = [x2s_norm[:,0:2]]
    print('x1.shape',x1.shape)

    X_3d = []
    for x1,x2 in zip(x1s,x2s):
        X_3d.append(traingulate_point(P1,P2,x1,x2))

    X_3d = [X_3d[:,0:2],X_3d[:,2]];
    result = X_3d;
    return result

def infront_of_camera(P, X):
    R,t = R_t_from_P(P)
    C = -R.T @ t.T
    l = (X - C) * R[2,:].T
    if l[2]>0:
        return True
    else:
        return False
    # X_p = P @ X
    # if X_p[2] > 0:
    #     return True
    # else:
    #     return False


def check_if_infront_of_both_cameras(P1,P2,cor):
    x1s = cor[:,0]
    x2s = cor[:,1]
    P2 = np.array(P2)

    result = 0
    X_homos = cv2.triangulatePoints(P1,P2,x1s.T,x2s.T).T
    for X_homo in X_homos:
        X = X_homo[:3] / X_homo[-1]

        if infront_of_camera(P1, X) and infront_of_camera(P2, X):
            result += 1

    return result

def essential_matrix_decomposition(E, cor):
    """
    Use this code to check correct:
    R1, R2, T = cv2.decomposeEssentialMat(E)
    print(R1)
    print(R1)
    print(T)"""
    # define camera 1 matrix
    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    # determine camera 2 matrix
    U,_,Vt = np.linalg.svd(E)
    D1 = np.eye(3)
    D1[2,2] = 0
    E = U @ D1 @ Vt

    U,_,Vt = np.linalg.svd(E)
    if(np.linalg.det(U @ Vt) < 0):
        Vt[:,2] = -Vt[:,2]

    u3 = U[:,2]
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    R1 = U @ W @ Vt
    if(np.linalg.det(R1) < 0):
        R1 = -R1

    R2 = U @ W.T @ Vt
    if(np.linalg.det(R2) < 0):
        R2 = -R2

    t1 = -u3
    t2 = u3

    src = cor[:,0]
    dst = cor[:,1]
    x1s = np.column_stack([src, np.ones(src.shape[0])])
    x2s = np.column_stack([dst, np.ones(dst.shape[0])])

    #for each possible R and t combo, check how many points are infront of both cameras
    count1 = check_if_infront_of_both_cameras(P1,combine_R_t(R1,t1),cor)
    count2 = check_if_infront_of_both_cameras(P1,combine_R_t(R1,t2),cor)
    count3 = check_if_infront_of_both_cameras(P1,combine_R_t(R2,t1),cor)
    count4 = check_if_infront_of_both_cameras(P1,combine_R_t(R2,t2),cor)

    print([count1,count2,count3,count4])

    max_count = np.max([count1,count2,count3,count4])
    if count1==max_count:
        P2 = combine_R_t(R1,t1)
    elif count2==max_count:
        P2 = combine_R_t(R1,t2)
    elif count3==max_count:
        P2 = combine_R_t(R2,t1)
    elif count4==max_count:
        P2 = combine_R_t(R2,t2)

    return P1, P2

class EssentialMatrixTransform(object):
  def __init__(self):
    self.params = np.eye(3)

  def __call__(self, coords):
    coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
    return coords_homogeneous @ self.params.T

  def estimate(self, src, dst):
    assert src.shape == dst.shape
    assert src.shape[0] >= 8

    # Setup homogeneous linear equation as dst' * F * src = 0.
    A = np.ones((src.shape[0], 9))
    A[:, :2] = src
    A[:, :3] *= dst[:, 0, np.newaxis]
    A[:, 3:5] = src
    A[:, 3:6] *= dst[:, 1, np.newaxis]
    A[:, 6:8] = src

    # Solve for the nullspace of the constraint matrix.
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)

    # Enforcing the internal constraint that two singular values must be
    # non-zero and one must be zero.
    U, S, V = np.linalg.svd(F)
    S[0] = S[1] = (S[0] + S[1]) / 2.0
    S[2] = 0
    self.params = U @ np.diag(S) @ V

    return True

  def residuals(self, src, dst):
    # Compute the Sampson distance.
    src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
    dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

    F_src = self.params @ src_homogeneous.T
    Ft_dst = self.params.T @ dst_homogeneous.T

    dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
                                       + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)


if __name__ == "__main__":
    cor = np.load('cor.npy')
    E = np.load('E.npy')
    essential_matrix_decomposition(E, cor)
