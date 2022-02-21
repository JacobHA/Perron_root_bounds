import numpy as np
from linalg_functions import Perron_info

def gamma_inf(A,B):
    N = A.shape[0]
    rA,x,y = Perron_info(A)

    result = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                result += x[i] * y[i] * (B[i][i] - A[i][i])
            else:
                result += (A[i][j] * x[i] * y[j] ) * np.log(B[i][j]/A[i][j])

    return rA + result
                
def main():
    dimension = 10
    A = np.random.rand(dimension,dimension)
    B = np.random.rand(dimension,dimension)
    print(f"Lower Bound on B's Perron root: {gamma_inf(A,B)}")
    print(f"Actual Perron root of B: {Perron_info(B)[0]}")
  
if __name__ == "__main__":
    main()
    

