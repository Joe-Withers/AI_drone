import numpy as np
from numpy.linalg import inv

"""
EKF notes:
state of world X (map) - includes: state of camera, position of landmarks
    modelled as multivariate gaussian distribution X ~ N(mu,epsilon)
    probability density function is p_X()
        d - num dimensions of mu - 13 + 3 * n, where n is the number of landmarks
    mu - state (mean) vector - includes: camera state and n landmarks 3d positions
    sigma - covariance matrix (d x d dimensions)
        starts uncorrelated i.e. eye(d,d)
    camera state - 13-dimension vector x_c
        x_c =  [r^W
                q^WR
                v^W
                w^R]
                r - 3D position
                q - orientation quaternion
                v - linear velocity
                w - angular velocity
        x =    [x_c
                y_1
                ...
                y_N]
        epsilon =   [epsilon-x_c-x_c epsilon-x_c-y_1 ... epsilon-x_c-y_N
                        ...             ...          ...         ...
                        ...             ...              epsilon-y_N-y_N]
"""
PI = 3.14
class Extended_Kalman_Filter():
    def __init__(self, dim_x, dim_z, dim_u=0):
        pass

    def p_X(self, X, mu, sigma, d):
        exp = (-0.5) * (X-mu).T * np.linalg.inv(sigma) * (X-mu)
        base = 1/(np.sqrt((2*PI)^d))
        return base**exp

    def f(self, mu, X_prev):
        """ state transition function. """
        pass

    def prediction_step(self, mu, X_prev, epsilon_prev):
        X_pred = f(mu, X_prev)
        epsilon_pred = F*epsilon_prev*F.T + Q

#******************************************************************************************
    def kf_predict(X, P, A, Q, B, U):
         X = np.dot(A, X) + np.dot(B, U)
         P = np.dot(A, np.dot(P, A.T)) + Q
         return X, P

     def kf_update(X, P, Y, H, R):
         IM = dot(H, X)
         IS = R + dot(H, dot(P, H.T))
         K = dot(P, dot(H.T, inv(IS)))
         X = X + dot(K, (Y-IM))
         P = P - dot(K, dot(IS, K.T))
         LH = gauss_pdf(Y, IM, IS)
         return (X,P,K,IM,IS,LH)

    def gauss_pdf(X, M, S):
         if M.shape()[1] == 1:
             DX = X - tile(M, X.shape()[1])
             E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
             E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
             P = exp(-E)
         elif X.shape()[1] == 1:
             DX = tile(X, M.shape()[1])- M
             E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
             E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
             P = exp(-E)
         else:
             DX = X-M
             E = 0.5 * dot(DX.T, dot(inv(S), DX))
             E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
             P = exp(-E)
         return (P[0],E[0])


if __name__ == "__main__":
    #time step of mobile movement
    dt = 0.1
    # Initialization of state matrices
    X = array([[0.0], [0.0], [0.1], [0.1]])
    P = diag((0.01, 0.01, 0.01, 0.01))
    A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,\
     1]])
    Q = eye(X.shape()[0])
    B = eye(X.shape()[0])
    U = zeros((X.shape()[0],1))
     - 5 -
    # Measurement matrices
    Y = array([[X[0,0] + abs(randn(1)[0])], [X[1,0] +\
     abs(randn(1)[0])]])
    H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
    R = eye(Y.shape()[0])
    # Number of iterations in Kalman Filter
    N_iter = 50
    # Applying the Kalman Filter
    for i in arange(0, N_iter):
     (X, P) = kf_predict(X, P, A, Q, B, U)
     (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
     Y = array([[X[0,0] + abs(0.1 * randn(1)[0])],[X[1, 0] +\
     abs(0.1 * randn(1)[0])]])
