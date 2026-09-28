import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        X = np.array(X)
        n = len(X)
        w = np.zeros(X.shape[1])
        b = 0
        i = 0
        
        while i < epochs:
            y_hat = X @ w + b
            Loss = (1/n) * sum((y_hat - y)**2)

            dL_dW = (2*X.T @ (y_hat - y))/n
            dL_dB = (2*np.sum(y_hat - y))/n

            w = w - lr*dL_dW
            b = b - lr*dL_dB
            i+=1
            
        return (np.round(w,5),np.round(b,5))
