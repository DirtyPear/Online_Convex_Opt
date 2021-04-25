import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm

data=pd.read_excel('48_Port_data.xlsx',index_col=0) # data are from Fama-French website
r=data.iloc[-240:,:]/100 # computing return

def get_value(w, r, cov, A, lam): # value of function f_t
    term1 = A*lam*norm.ppf(0.95)*np.sqrt(w.dot(cov).dot(w))
    term2 = A*r.dot(w)
    return term1-term2


def get_derivative(w, r, cov, A, lam): # value of derivative of f_t
    term1 = A*lam*norm.ppf(0.95)*np.power(w.dot(cov).dot(w), -1/2)*cov.dot(w)
    term2 = A*r
    return term1-term2


def projection(y): # projection on set K
    def func(x):
        return ((x-y)**2).sum()

    def con():
        cons=({'type':'eq', 'fun': lambda x: x.sum()-1})
        return cons

    bound=Bounds([0]*len(y),[1]*len(y))
    res=minimize(func,np.array([1/n]*n),method='SLSQP',constraints=con(), bounds=bound)
    
    return res.x
  
n=r.shape[1] # number of assets in the portfolio
A=1 # initial wealth

w_equal=pd.DataFrame(np.ones((n,len(r)))/n,columns=range(1,len(r)+1))

def simulation(A, lam, step_size): 
    # apply Online Gradient Descent with fixed step size
    # return the value of the portfolio through the testing period
    w=pd.DataFrame(index=r.columns)
    w[1]=np.array([1/n]*n) 
    A_df=pd.Series(index=r.index)#, columns=['Portfolio Value'])
    A_df[0]=A
    for t in range(1,len(r)):
        r_t=r.iloc[t,:]
        cov_t=r.iloc[:t+1,:].cov()
        A_t=A_df[t-1]*(1+r_t.dot(w[t]))
        A_df[t]=A_t
        y_next=w[t]-step_size*get_derivative(w[t], r_t, cov_t, A_t, lam)
        w[t+1]=projection(y_next)
    return A_df, w

# try different lambda  
A1,w1 = simulation(A, 1, 0.1)
A2,w2 = simulation(A, 10, 0.1)  
A3,w3 = simulation(A, 100, 0.1) 
A4,w4 = simulation(A, 1000, 0.1)
A5,w5 = simulation(A, -1, 0.1)
A6,w6 = simulation(A, -10, 0.1)

A_collect=pd.concat([A1,A2,A3,A4,A5,A6],axis=1)
A_collect.columns=['1','10','100','1000','-1','-10']

def simulation_dyn_step(A, lam):
    # apply Online Gradient Descent with dynamic step size
    # return the value of the portfolio through the testing period
    w=pd.DataFrame(index=r.columns)
    w[1]=np.array([1/n]*n) 
    A_df=pd.Series(index=r.index)#, columns=['Portfolio Value'])
    A_df[0]=A
    for t in range(1,len(r)):
        step_size=1/((100*lam*norm.ppf(0.95)-1)*np.sqrt(t))
        r_t=r.iloc[t,:]
        cov_t=r.iloc[:t+1,:].cov()
        A_t=A_df[t-1]*(1+r_t.dot(w[t]))
        A_df[t]=A_t
        y_next=w[t]-step_size*get_derivative(w[t], r_t, cov_t, A_t, lam)
        w[t+1]=projection(y_next)
    return A_df, w

# try different step sizes including dynamic step size
A7,w7 = simulation_dyn_step(A, 0.1)
A8,w8 = simulation(A, 1, 1)
A9,w9 = simulation(A, 1, 0.1)
A10,w10 = simulation(A, 1, 0.01)
A11,w11 = simulation(A, 1, 0.001)

A_collect_2=pd.concat([A7,A8,A9,A10,A11],axis=1)
A_collect_2.columns=['dynamic','1','0.1','0.01','0.001']

# equally weighted portfolio as comparison
A_eq=pd.Series(index=r.index)
A_eq[0]=A
for t in range(1, len(r)):
    r_t=r.iloc[t,:]
    A_eq[t]=A_eq[t-1]*(1+r_t.dot(np.array([1/n]*n)))

# Shapre ratio
print((A_collect.iloc[-1,:]-1)/A_collect.std())
print((A_collect_2.iloc[-1,:]-1)/A_collect_2.std())
print((A_eq[-1]-1)/A_eq.std()) 
