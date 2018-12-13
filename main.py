import numpy as np


def calculate_h(p,p_,homo_representation):
    "takes setof points p and their corresponting points p_ , calculates the Homography matrix, returns 3x3 matrix"
    if(homo_representation):
        A = np.zeros([3*p.shape[0],8])
        for i in p.shape[0]:
            for j in range(0,3): 
                if(i*3+j != 2 ):
                    A[i*3+j][j*3:j*3+3] = p
                else:
                    A[i*3+j][0] = p[0]
                    A[i*3+j][1] = p[1]
        
        B = p
        
        H = np.linalg.lstsq(A,B)
        
        return H