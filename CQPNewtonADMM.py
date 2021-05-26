# -*- coding: utf-8 -*-
import numpy as np

# Solves
#        min 1/2x^TPx + q^Tx
#        s.t Ax >= b
# Pass data (P,q,A,b) to CQP_NewtonADMM

#Objective function
def quadratic(x,P,q):
    return(1/2*x.T@P@x+q.T@x)

#Computes ADMM residual
def ADMM_residual(M,q,A,b,x,y,z,mu):
    res1 = M@x - mu*A.T@(y-z)+q
    Ax   = A@x
    res2 = y - np.maximum(Ax+z,b)
    res3 = y- Ax
    return np.concatenate([res1,res2,res3])

#Computes norm of ADMM residual
def norm_ADMM_residual(M,q,A,b,x,y,z,dx,dy,dz,alpha,mu):
     ares = ADMM_residual(M,q,A,b,x + alpha*dx,y + alpha*dy,z + alpha*dz,mu)
     return np.linalg.norm(ares)

#Performs a simple backtracking linesearch enforcing descent
def backtracking(M,q,A,b,x,y,z,dx,dy,dz,mu):
    alpha = 1.0
    c     = 1.0E-05
    beta  = 0.5
    r0    = norm_ADMM_residual(M,q,A,b,x,y,z,dx,dy,dz,0.0,mu)
    while norm_ADMM_residual(M,q,A,b,x,y,z,dx,dy,dz,alpha,mu) > (1-c*alpha)*r0:
        alpha *= beta
    return alpha

#Computes a matrix element of the generalized Jacobian of the ADMM residual
def ADMM_jacobian(M,A,b,x,y,z,mu):
    m = A.shape[0]
    AT = A.T
    Ax = A@x
    d  = np.ones(m)
    d[np.where(Ax+z-b <= 0)]    = 0.0
    dA = A.copy()
    dA[np.where(Ax+z-b <= 0),:] = 0.0
    dres1 = np.hstack([M,-mu*AT,mu*AT])
    dres2 = np.hstack([-dA,np.eye(m),-np.diag(d)])
    dres3 = np.hstack([-A,np.eye(m),np.zeros((m,m))])
    return np.vstack([dres1,dres2,dres3])
    
#The solver itself
def CQP_NewtonADMM(data, mu=10, itermax = 100, primal_tol=1e-8, dual_tol=1e-8, gamma = 1e-8):
    #Initializing problem data and dimensions
    (P,q,A,b) = data
    AT = A.T
    M  = P+mu*AT@A
    m  = A.shape[0]
    n  = A.shape[1]
    #Initial iterate
    x = np.zeros(n)
    y = np.zeros(m)
    z = np.zeros(m)
    #initialize stopping criterion
    i = 0
    normAT = np.linalg.norm(AT, ord='fro')
    tol = np.minimum(primal_tol, dual_tol/(2*mu*normAT))
    while np.linalg.norm(ADMM_residual(M,q,A,b,x,y,z,mu)) > tol and i < itermax:
        resid = ADMM_residual(M,q,A,b,x,y,z,mu)
        J   = ADMM_jacobian(M,A,b,x,y,z,mu)
        JT    = J.T
        JTJ   = JT@J
        #Tikhonov regularized Newton step with scaling Tikhonov matrix diag(J^TJ) and penalty gamma>0
        dsol  = -np.linalg.solve(JTJ+gamma*np.diag(np.diagonal(JTJ)),JT@resid)
        dx    = dsol[:n]
        dy    = dsol[n:n+m]
        dz    = dsol[n+m:]
        alpha = backtracking(M,q,A,b,x,y,z,dx,dy,dz,mu)
        x = x + alpha*dx
        y = y + alpha*dy
        z = z + alpha*dz
        i += 1
    #reporting
    if np.linalg.norm(ADMM_residual(M,q,A,b,x,y,z,mu))>tol:
        print("ADMM did not converge to a solution within the given tolerance")
        print("Norm of the ADMM residual=",np.linalg.norm(ADMM_residual(M,q,A,b,x,y,z,mu)))
        print("Change either itermax or gamma!")
    else: 
        print("ADMM converged successfully")
        print("Norm of the ADMM residual =",np.linalg.norm(ADMM_residual(M,q,A,b,x,y,z,mu)))
        print("Iterations required =",i)
        print("Optimal value =",quadratic(x,P,q))
    return(x)