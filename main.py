import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy.linalg import inv
from scipy.interpolate import RectSphereBivariateSpline as interpolate
import math
import random


def calculate_h(p,p_):
    "takes setof points p and their corresponting points p_ , calculates the Homography matrix, returns 3x3 matrix"
    B=p_
    
    A = np.zeros([2*p.shape[0],8])  #construct A from input points
    for i in range(p.shape[0]): 
        A[i*2] = p[i,0],p[i,1],1,0,0,0,-p[i,0]*p_[i,0],-p[i,1]*p_[i,1]    
        A[i*2+1] = 0,0,0, p[i,0],p[i,1],1,-p[i,0]*p_[i,0],-p[i,1]*p_[i,1] 
#            for j in range(3): 
#                if((i*3+j)%3 != 2 ):
#                    A[i*3+j][j*3:(j*3)+2] = p[i][0],p[i][1]
#                else:
#                    A[i*3+j][j*3] = p[i][0]
#                    A[i*3+j][j*3+1] = p[i][1]
#         B=np.pad(B,(0,1),'constant',constant_values=1) #pad with ones to make it homogoneous
#         B=B[0:-1,:]
         
         
    
    B=B.flatten().reshape(-1,1) # flatten and reshape to be one column for dimension suitability
    
    H = np.linalg.lstsq(A,B,rcond=None)[0] #returns H
    
    H=np.append(H,1)  #puts 1 at the end of H
   
    H = np.reshape(H,[3,3])
    return H


def get_correspondance_auto(image1_gray,image2_gray):
    
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1_gray,None)
    kp2, des2 = orb.detectAndCompute(image2_gray,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    p=[]
    p_ = []
    k=[]
    k2=[]
    for match in matches:
        index1 = match.trainIdx
        p.append(kp1[index1].pt)
        index2 = match.queryIdx
        p_.append(kp2[index2].pt)
        k.append(kp1[index1])
        k2.append(kp2[index2])
        
    
    matchImg = cv.drawMatches(image1_gray,kp1,image2_gray,kp2,matches,image2_gray)
    cv.imwrite('Matches.png', matchImg)
    
    p=np.array(p)
    p_=np.array(p_)
    return p, p_



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


#def get_correspondance_manually(number_of_points):
#    return  plt.ginput(number_of_points*2)
# 
    
def ransac_distance(single_p,single_p_,h):
    point = np.array([single_p[0],single_p[1],1])
    calculated_point_ = np.dot(h,point.reshape(-1,1))
    calculated_point_ /= calculated_point_[2]
    
    error = calculated_point_[0:2] - single_p_.reshape(-1,1) 
   
    error = np.square(error)
    
    error = np.sum(error)
    
    return math.sqrt(error)

def ransac(p,p_,threshold,iterations):
    
    max_inliners = 0 
    best_h = None
    best_error = 10000000
    for i in range(iterations):
        inliners = 0 
        
        
        
        randp  = np.zeros([4,2]) 
        randp_ = np.zeros([4,2])
        
        for j in range(4):
            random_index = random.randrange(0, p.shape[0])
            #print(random_index)
            randp[j]=p[random_index]
            randp_[j]=p_[random_index]
            
        H = calculate_h(randp,randp_)
        
       
        
        for j in range(p.shape[0]):
            error = ransac_distance(p[j],p_[j],H)
           
            
            if(error<threshold):
                inliners+=1
            
            
        if(inliners>max_inliners):
            max_inliners = inliners
            best_h = H
            
    return best_h , max_inliners 
            
    
    

image1 = cv.imread("./image1.png")
image2 = cv.imread("./image2.png")

image1_gray = cv.cvtColor(image1,cv.COLOR_RGB2GRAY)
image2_gray = cv.cvtColor(image2,cv.COLOR_RGB2GRAY)
p,p_ = get_correspondance_auto(image1_gray,image2_gray)
# Initiate FAST object with default values
# Initiate SIFT detector

#sift = cv.xfeatures2d.SIFT_create()
#
## find the keypoints and descriptors with SIFT
#kp1, des1 = sift.detectAndCompute(image1,None)
#kp2, des2 = sift.detectAndCompute(image2,None)
## create BFMatcher object




H = calculate_h(p,p_)


ransac_h, inliners = ransac(p,p_,20,250)
print(inliners)

inverse_warp(ransac_h,image1,image2)
#inverse_warp(H,image1,image2)

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


