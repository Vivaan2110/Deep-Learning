import numpy as np 

def looped_positional_encoding(position:int, d_model:int):
    PE=np.zeros(d_model)
    for i in range(0,d_model,2):
        divide=np.power(10000,i,d_model)
        PE[i]=np.sin(position/divide)
        if i+1<position:
            PE[i+1]=np.cos(position/divide)
    
    return PE

def vectorial_positiona_encoding(position: int, d_model:int):
    i=np.arrange(d_model) # Creates indices from 0 to d_model-1
    
    angle_rates=1/np.power(10000,2*(i//2)/d_model)
    angles=position*angle_rates
    
    PE=np.zeros(d_model)
    PE[0::2]=np.sin(angles[0::2]) # All even indices are sin
    PE[1::2]=np.cos(angles[1::2]) # All odd indices are odd
    
    return PE