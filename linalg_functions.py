import numpy as np

def DOMeigs(M):
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
