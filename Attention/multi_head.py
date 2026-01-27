import numpy as np 

def softmax(x):
    x=x-np.max(x, axis=-1, keepdims=True)
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def multi_head_attention(X, W_k, W_q, W_v, W_o, h):
    n, d_model=X.shape
    d_k=d_v=int(d_model/h)
    
    Q=np.matmul(X, W_q) # n, d_model
    K=np.matmul(X, W_k)
    V=np.matmul(X, W_v)
    
    Q=Q.reshape(n, h, d_k)
    K=K.reshape(n, h, d_k)
    V=V.reshape(n, h, d_k)
    
    heads=[]
    
    for i in range(h):
        S=np.matmul(Q[:,i,:],np.transpose(K[:,i,:]))
        S_norm=S/np.sqrt(d_k)
        head=np.matmul(softmax(S_norm), V[:,i,:])
        heads.append(head)
    
    return np.matmul(np.concatenate(heads, axis=-1), W_o) # n,d_model