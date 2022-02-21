import numpy as np
from linalg_functions import Perron_info


def bapat(A,B):
  N = A.shape[0]
  rA, uA, vA = Perron_info(A)
  for i in range(N):
      for j in range(N):
          prod *= (B[i][j]/A[i][j])**(A[i][j]*uA[i]*vA[j]/rA)
  return rA*prod
        
def main():
    A = np.random.rand(dimension,dimension)
    B = np.random.rand(dimension,dimension)
    rB,_,_ = Perron_info(B)
    print(f"Lower bound on Perron root of B: {bapat(A,B)}")
    print(f"True Perron root of B: {rB}")
          
if __name__ == "__main__":
    main()
