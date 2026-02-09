import numpy as np 

# Want to create a matrix where the lower traingular part is 0 and upper triangular part is -infinfty 
def unfinished_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))    

def softmax(x):
    x=x-np.max(x,axis=-1, keepdims=True)
    exp_x=np.exp(x)
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)

def masked_self_attention(X:np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray)->np.ndarray:
    Q=np.dot(X,W_q)
    K=np.dot(X,W_k)
    V=np.dot(X,W_v)
    
    seq_len, d_m=Q.shape
    
    scores=np.matmul(Q, np.transpose(K))/np.sqrt(d_m)
    
    mask=unfinished_mask(seq_len)
    
    mask=np.where(mask==0, -1e10, scores) # If mask==0 is true, select 1e10, if not select value from scores itself
    
    weights=softmax(mask)
    
    return np.matmul(weights, V)