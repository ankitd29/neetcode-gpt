import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        n = len(x)
        mean = np.mean(x)
        variance = np.var(x)

        x_hat = (x - mean)/(variance + 10**-5)**0.5
        y = x_hat * gamma + beta
        return np.round(y, 5)
