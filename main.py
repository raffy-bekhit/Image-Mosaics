import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy.linalg import inv
from scipy.interpolate import RectSphereBivariateSpline as interpolate
import math



def calculate_h(p,p_,homo_representation=True):
    "takes setof points p and their corresponting points p_ , calculates the Homography matrix, returns 3x3 matrix"
    B=p_
    if(homo_representation): # a flag indicates whether the given points are hogoneous representation or not
        A = np.zeros([3*p.shape[0],8]) 
        for i in range(0,p.shape[0]):
            for j in range(0,3): 
                if((i*3+j)%3 != 2 ):
                    A[i*3+j][j*3:(j*3)+3] = p[i,0:3]
                else:
                    A[i*3+j][j*3] = p[i,0]
                    A[i*3+j][j*3+1] = p[i,1]
                    
       
    else:
         A = np.ones([3*p.shape[0],8])  #construct A from input points
         for i in range(p.shape[0]): 
            for j in range(3): 
                if((i*3+j)%3 != 2 ):
                    A[i*3+j][j*3:(j*3)+2] = p[i][0],p[i][1]
                else:
                    A[i*3+j][j*3] = p[i][0]
                    A[i*3+j][j*3+1] = p[i][1]
         B=np.pad(B,(0,1),'constant',constant_values=1) #pad with ones to make it homogoneous
         B=B[0:-1,:]
         
         
     
    B=B.flatten().reshape(-1,1) # flatten and reshape to be one column for dimension suitability
    
    H = np.linalg.lstsq(A,B,rcond=None)[0] #returns H
    H=np.append(H,1)  #puts 1 at the end of H
   
    H = np.reshape(H,[3,3])
    return H


def inverse_warp(H,image1, image2):
    warped = image2
    H_inv = inv(H)
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            position = np.matmul(H_inv,np.array([i,j,1]).reshape(-1,1))
            x=position[0]/position[2]
            y=position[1]/position[2]
           
            print(x)
            x1 = math.floor(x)
            x2 = math.ceil(x)
            
            y1=math.floor(-y)
            y2 = math.ceil(-y)
            print(x1,x2,y1,y2)
#            if(x1!=x2):
#                xx=[x1,x2]
#            else:
#                xx=[x1]
#            if(y1!=y2):
#                yy=[y1,y2]
#            else:
#                yy=[y1]
            
            
            
            xx,yy = np.meshgrid([x1,x2],[y1,y2])
           
            
            for rgb in range(3):
                
                interpolator = interpolate(np.array([x1,x2]),np.array([y1,y2]),np.array([[image1[x1,y1,rgb],image1[x1,y2,rgb]],[image1[x2,y1,rgb],image1[x2,y2,rgb]]]))
            
    
    return warped


def get_correspondance_manually(number_of_points):
    return  plt.ginput(number_of_points*2)
    
#
#p =np.array( [(1,2),(3,2),(5,2),(9,2)])
#p_ = np.array( [(44,2),(55,2),(99,2),(2,2)])
#
#
##p = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
##p_= np.array([[2,2,1],[2,2,1],[2,2,1],[2,2,1]])
##
##
#H = calculate_h(np.asarray(p),np.asarray(p_),False)
#print("H= ",H)


image1 = cv.imread("./image1.png")
image2 = cv.imread("./image2.png")

# Initiate FAST object with default values
# Initiate SIFT detector

#sift = cv.xfeatures2d.SIFT_create()
#
## find the keypoints and descriptors with SIFT
#kp1, des1 = sift.detectAndCompute(image1,None)
#kp2, des2 = sift.detectAndCompute(image2,None)
## create BFMatcher object



orb = cv.ORB_create()
image1_gray = cv.cvtColor(image1,cv.COLOR_RGB2GRAY)
image2_gray = cv.cvtColor(image2,cv.COLOR_RGB2GRAY)

kp1, des1 = orb.detectAndCompute(image1_gray,None)
kp2, des2 = orb.detectAndCompute(image2_gray,None)



bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)


# Match descriptors.
matches = bf.match(des1,des2)


matches = sorted(matches, key = lambda x:x.distance)


p=[]
p_ = []
    
for i in range (100):
    index1 = matches[i].trainIdx
    p.append(kp1[index1].pt)
    index2 = matches[i].queryIdx
    p_.append(kp2[index2].pt)



p=np.array(p)
p_=np.array(p_)


H = calculate_h(p,p_,False)

inverse_warp(H,image1,image2)

#pro = np.matmul(H,np.array([531,508,1]).reshape(-1,1))

# Sort them in the order of their distance.
#fig = plt.figure(figsize=(1,2))

#fig.add_subplot(1,1,1);
#plt.figure(1)
#plt.imshow(image1,cmap='gray');
#
#n=plt.ginput(2)
#
#plt.show()
#
#print(n)
cv.waitKey(0); #press key to close image
cv.destroyAllWindows(); #destroys all windows
#
#
#


