import numpy as np


def DOMeigs(M):
    # returns dominant eigen-information for matrix M
    vals,vecs = np.linalg.eig(M) # Eigen
    vecs = (vecs.T).real # format and realize
    dict_of_eig = dict(zip(vals,vecs))
    dom_eig_val=sorted(dict_of_eig)[-1]
    dom_eig_vec=dict_of_eig[dom_eig_val]
    
    assert dom_eig_val == dom_eig_val.real, ("Complex Perron root!")
    return dom_eig_val.real, dom_eig_vec

def Perron_info(M):
    # returns rho, u, v
    rho,v = DOMeigs(M)
    rho,u = DOMeigs(M.T)
    v /= sum(v)
    u /= u.dot(v)
    return rho, u, v
  
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
    return gamma_inf(A,B)
  
if __name__ == "__main__":
    main()
    
