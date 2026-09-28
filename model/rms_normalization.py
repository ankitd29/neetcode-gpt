import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        RMSx = np.sqrt(np.mean(x**2) + eps)
        x_hat = x/RMSx
        output = gamma*x_hat
        return np.round(output,4).tolist()
