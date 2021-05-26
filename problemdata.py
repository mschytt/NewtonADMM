# -*- coding: utf-8 -*-
import numpy as np

# Constructs the problem data also used in the thesis
# When multiple problems of the same dimensions are used, we count up the rng seed


# CQP_problem constructs CQP problem data
#        min 1/2x^TPx + q^Tx
#        s.t Ax >= b
def CQP_problem(m,n):
    rng = np.random.default_rng(65687777)
    P   = rng.random((n,n))
    P   = P.T@P+n*np.eye(n)
    q   = rng.random(n)
    A   = rng.random((m,n))
    b   = rng.random(m)
    return(P,q,A,b)

# Lasso_problem constructs ols problem data
#        min 1/2||Ax-b||_2^2
def Lasso_problem(n,m):
    rng = np.random.default_rng(65687777)
    A   = rng.random((m,n))
    b   = rng.random(m)
    return(A,b)

