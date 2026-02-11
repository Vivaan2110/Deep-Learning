import numpy as np 

def layer_norm(X:np.ndarray, gamma:np.ndarray, beta:np.ndarray, epsilon=1e-5)->np.ndarray:
    mean=np.mean(X, axis=-1, keepdims=True)
    var=np.var(X, axis=-1, keepdims=True)
    
    x_hat=(X-mean)/np.sqrt(var+epsilon)
    
    layerNorm=gamma*x_hat+beta
    
    return layerNorm