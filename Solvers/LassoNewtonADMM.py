# -*- coding: utf-8 -*-
import numpy as np
import pywt

# Solves
#        min 1/2||Ax-b||_2^2 + lmbd||x||_1
# Pass data (A,b,lmbd) to Lasso_NewtonADMM

#Objective function
def lasso(A,b,x,lmbd):
    return 0.5  * np.linalg.norm(A.dot(x)-b)**2 \
          +lmbd * np.linalg.norm(x,ord=1)

#Softthresholding function
def soft_threshold(x,lmbd):
    return pywt.threshold(x,lmbd)

#Computes ADMM residual
def ADMM_residual(M,A,b,lmbd,x,y,z,mu):
    ATb  = A.T@b
    res1 = M@x - mu*(y-z) - ATb
    res2 = y - soft_threshold(x+z,lmbd/mu)
    res3 = y - x
    return np.concatenate([res1,res2,res3])

#Computes norm of ADMM residual
def norm_ADMM_residual(M,A,b,lmbd,x,y,z,dx,dy,dz,alpha,mu):
     ares = ADMM_residual(M,A,b,lmbd,x + alpha*dx,y + alpha*dy,z + alpha*dz,mu)
     return np.linalg.norm(ares)

#Performs a simple backtracking linesearch enforcing descent
def backtracking(M,A,b,lmbd,x,y,z,dx,dy,dz,mu):
    alpha = 1.0
    c     = 1.0E-05
    beta  = 0.5
    r0    = norm_ADMM_residual(M,A,b,lmbd,x,y,z,dx,dy,dz,0.0,mu)
    while norm_ADMM_residual(M,A,b,lmbd,x,y,z,dx,dy,dz,alpha,mu) > (1-c*alpha)*r0:
        alpha *= beta
    return alpha

#Computes a matrix element of the generalized Jacobian of the ADMM residual
def ADMM_jacobian(M,lmbd,x,y,z,mu):
    n = M.shape[0]
    dres1 = np.hstack([M,-mu*np.eye(n),mu*np.eye(n)])
    d     = np.ones(n)
    d[np.where(np.abs(x+z)<lmbd/mu)] = 0.0
    dres2 = np.hstack([-np.diag(d),np.eye(n),-np.diag(d)])
    dres3 = np.hstack([-np.eye(n),np.eye(n),np.zeros((n,n))])
    return np.vstack([dres1,dres2,dres3])
    
#The solver itself
def Lasso_NewtonADMM(data, mu=0.1, itermax = 100, primal_tol=1e-6, dual_tol=1e-6, gamma = 1e-8):
    #Initializing problem data and dimensions
    (A,b,lmbd) = data
    n  = A.shape[1]
    AT = A.T
    M  = AT@A + mu*np.eye(n)
    #Initial iterate
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    #initialize stopping criterion
    i = 0
    normA = np.linalg.norm(A, ord='fro')
    tol = np.minimum(primal_tol, dual_tol/(2*normA*mu))
    while np.linalg.norm(ADMM_residual(M,A,b,lmbd,x,y,z,mu)) > tol and i < itermax:
        resid = ADMM_residual(M,A,b,lmbd,x,y,z,mu)
        J   = ADMM_jacobian(M,lmbd,x,y,z,mu)
        JT    = J.T
        JTJ   = JT@J
        #Tikhonov regularized Newton step with scaling Tikhonov matrix diag(J^TJ) and penalty gamma>0
        dsol  = -np.linalg.solve(JTJ+gamma*np.diag(np.diagonal(JTJ)),JT@resid)
        dx    = dsol[:n]
        dy    = dsol[n:2*n]
        dz    = dsol[2*n:]
        alpha = backtracking(M,A,b,lmbd,x,y,z,dx,dy,dz,mu)
        x = x + alpha*dx
        y = y + alpha*dy
        z = z + alpha*dz
        i += 1
    #reporting
    if np.linalg.norm(ADMM_residual(M,A,b,lmbd,x,y,z,mu))>tol:
        print("ADMM did not converge to a solution within the given tolerance")
        print("Norm of the ADMM residual=",np.linalg.norm(ADMM_residual(M,A,b,lmbd,x,y,z,mu)))
        print("Change either itermax or gamma!")
    else: 
        print("ADMM converged successfully")
        print("Norm of the ADMM residual =",np.linalg.norm(ADMM_residual(M,A,b,lmbd,x,y,z,mu)))
        print("Iterations required =",i)
        print("Optimal value =",lasso(A,b,x,lmbd))
    return(x)