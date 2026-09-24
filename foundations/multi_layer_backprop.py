import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        z1 = np.dot(W1,x) + b1
        a1 = np.maximum(0,z1)
        z2 = np.dot(W2,a1) + b2
        y_hat = z2

        Loss = np.mean((y_hat - y_true)**2)
        
        dL_dyhat = 2*(y_hat - y_true)
        dW2 = dL_dyhat.reshape(-1,1)*a1.reshape(1,-1)
        db2 = dL_dyhat

        da1 = np.dot(np.transpose(W2),dL_dyhat)
        dz1 = da1* (z1 > 0).astype(float)
        dW1 = np.outer(dz1,np.transpose(x))
        db1 = dz1
        return {"loss": round(float(Loss),4),
        "dW1": np.round(dW1,4).tolist(), 
        "db1": np.round(db1,4).tolist(), 
        "dW2": np.round(dW2,4).tolist(), 
        "db2": np.round(db2,4).tolist(),
        }
