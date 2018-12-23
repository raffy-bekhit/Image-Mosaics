import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from numpy.linalg import inv
from scipy.interpolate import RectBivariateSpline as interpolate
import math
import random

from PIL import Image;


CORRESPONDENCE_AUT0 = 0 
CORRESPONDENCE_MANUAL = 1
CALCULATE_H_NORMALLY = 0
CALCULATE_H_RANSAC = 1


def calculate_h(p,p_):
    "takes setof points p and their corresponting points p_ , calculates the Homography matrix, returns 3x3 matrix"
    #p is array of correspondance points of image 1, p_ is same for image 2 
    B=p_
    A = np.zeros([2*p.shape[0],8])  #construct A from input points
    for i in range(p.shape[0]): #constructs A from given points of image 1 
        A[i*2] = p[i,0],p[i,1],1,0,0,0,-p[i,0]*p_[i,0],-p[i,1]*p_[i,0]    
        A[i*2+1] = 0,0,0, p[i,0],p[i,1],1,-p[i,0]*p_[i,1],-p[i,1]*p_[i,1] 

    
    B=B.flatten().reshape(-1,1) # flatten and reshape to be one column for dimension suitability
    
    H = np.linalg.lstsq(A,B,rcond=None)[0] #returns H
    
    H=np.append(H,1)  #puts 1 at the end of H
   
    H = np.reshape(H,[3,3]) #return H as matrix of shape 3x3
    return H


def get_correspondance_auto(image1_gray,image2_gray):
    "get coresspondance points between two given images by sift"
    orb = cv.ORB_create() 
    kp1, des1 = orb.detectAndCompute(image1_gray,None)#keypoints and descriptors of first image
    kp2, des2 = orb.detectAndCompute(image2_gray,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) #creates a matcher

    # Match descriptors.
    matches = bf.match(des1,des2) #matches the two descriptors
    matches = sorted(matches, key = lambda x:x.distance) #sorts matches where best matcvhes come first
    
    p=[] #list of correspondance point in first image
    p_ = [] #list of correspondance point in second image
    
    for match in matches: #loop on matches and fills p and p_
        index1 = match.queryIdx
        p.append((int(kp1[index1].pt[0]),int(kp1[index1].pt[1])) )
        
        
        index2 = match.trainIdx
        p_.append((int(kp2[index2].pt[0]),int(kp2[index2].pt[1])))
        
        
    
    #matchImg = cv.drawMatches(image1_gray,kp1,image2_gray,kp2,matches[0:20],image2_gray) #draws the best 20 matches on the image and saves it as output image
    #cv.imwrite('Matches.png', matchImg)
    
    p=np.array(p)
    p_=np.array(p_)
    return p, p_

 


def get_correspondance_manually(image1,image2,number_of_points):
		# Display images, select matching points
        fig = plt.figure()
        figA = fig.add_subplot(1,2,1)
        figB = fig.add_subplot(1,2,2)
	# Display the image
	# lower use to flip the image
        figA.imshow(image1)#,origin='lower')
        figB.imshow(image2)#,origin='lower')
        plt.axis('image')
	# n = number of points to read
        p1 = np.zeros([(number_of_points//2),2])
        p2 = np.zeros([number_of_points//2,2])
        pts = plt.ginput(n=number_of_points, timeout=0)
        p1_itr = 0
        p2_itr = 0
        for i in range(0, number_of_points):
            if (i % 2 == 0):
                p1[p1_itr] = pts[i]
                print("p1 of index ", p1_itr, " is ", pts[i])
                p1_itr += 1

            else:
                p2[p2_itr] = pts[i]
                print("p2 of index ", p2_itr, " is ", pts[i])
                p2_itr += 1
        return p1, p2


def InvTransform(h,point):
    "for a point in second image gets its correspondance in first image"

    inverse_point = np.dot(np.linalg.inv(h), point); #matrix multiplication with the inverse of H
    inverse_point[0] /= inverse_point[2] #divides by weight
    inverse_point[1] /= inverse_point[2]
    return inverse_point[0:2]

def transform(h,point):
    "for a point in first image gets its correspondance in second image "
    corr_point = np.dot(h, point);
    corr_point[0] /= corr_point[2];
    corr_point[1] /= corr_point[2]
    return corr_point[0:2]

def from_homo(points):
    "converts homogoneous representation of a point a non-homo representation"
    
    points[:,0] /= points[:,2]
    points[:,1] /= points[:,2]
   
    return points[:,0:2]
    
def warp(source_image,dest_image,h):
    "aligns the two images"
    source_height = source_image.shape[0]
    source_width = source_image.shape[1] 
    source_edges = np.array([[0,0],[source_width-1, 0],[0, source_height -1],[source_width-1, source_height-1]]) # array containing corner points of source image
    source_edges= np.pad(source_edges,(0,1),'constant',constant_values=1)     #pad 1 to make it homogoneous representation
    source_edges = source_edges[:-1] #remove a redudant row from padding

    corr_source_edges = from_homo(np.dot(h,source_edges.T).T) #get the points cerresponding to edges in non-homogoneous form each row is a point and each colum is x or y 
    
    #get the corresponting positions of the corner points
    max_mapped_i,max_mapped_j = int(np.ndarray.max(corr_source_edges[:,1],axis=0)),int(np.ndarray.max(corr_source_edges[:,0],axis=0)) 
    min_mapped_i,min_mapped_j =int( np.ndarray.min(corr_source_edges[:,1],axis=0)),int(np.ndarray.min(corr_source_edges[:,0],axis=0))
        
    
    #gets the size of source image after warping
    mapped_source_height = max_mapped_i-min_mapped_i+1  
    mapped_source_width = max_mapped_j-min_mapped_j+1
    
    #determines the shift that happens to the second image
    shiftHeight = -min_mapped_i
    shiftWidth = - min_mapped_j
    
    # new image containing the first image after warping
    mapped_source_image = np.zeros((mapped_source_height,mapped_source_width,3), dtype=np.uint8);

    
    #forward warping
    for i in range(0,source_height):
        for j in range(0, source_width):
            mapped_position = transform(h,np.array([j,i,1]))
            mapped_j = int(mapped_position[0])
            mapped_i = int(mapped_position[1])
           
            mapped_source_image[mapped_i+shiftHeight][mapped_j+shiftWidth] = source_image[i][j];
            
            
    #save forward warped image
    #the resulting image contains holes
    ###cv.imwrite("with holes.png",mapped_source_image)
    #inverse warping to remove holes
    for i in range(0, mapped_source_height):
        for j in range(0, mapped_source_width):
			#check if pixel is black
            if (int(mapped_source_image[i][j][0]) == 0 and int(mapped_source_image[i][j][1]) == 0 and int(mapped_source_image[i][j][2]) == 0):
				
                inverse_mapped_position = InvTransform(h,np.array([j - shiftWidth,i - shiftHeight,1]))
                inverse_mapped_i = inverse_mapped_position[1]
                inverse_mapped_j = inverse_mapped_position[0]
                if inverse_mapped_i <= source_height-1 and  inverse_mapped_i >= 0 and inverse_mapped_j <= source_width-1 and inverse_mapped_j >= 0:
					#interpolate
                    low_i = int(inverse_mapped_i);
                    low_j = int(inverse_mapped_j);
                    idistance = inverse_mapped_i - low_i;
                    jdistace = inverse_mapped_j - low_j;
                    
                    #interpolation for each channel
                    mapped_source_image[i][j][0] = (1-idistance)*(1-jdistace)*source_image[low_i][low_j][0] + (1-idistance)*(jdistace)*source_image[low_i][low_j+1][0] + (idistance)*(1-jdistace)*source_image[low_i+1][low_j][0] + (idistance)*(jdistace)*source_image[low_i+1][low_j+1][0]
                    mapped_source_image[i][j][1] = (1-idistance)*(1-jdistace)*source_image[low_i][low_j][1] + (1-idistance)*(jdistace)*source_image[low_i][low_j+1][1] + (idistance)*(1-jdistace)*source_image[low_i+1][low_j][1] + (idistance)*(jdistace)*source_image[low_i+1][low_j+1][1]
                    mapped_source_image[i][j][2] = (1-idistance)*(1-jdistace)*source_image[low_i][low_j][2] + (1-idistance)*(jdistace)*source_image[low_i][low_j+1][2] + (idistance)*(1-jdistace)*source_image[low_i+1][low_j][2] + (idistance)*(jdistace)*source_image[low_i+1][low_j+1][2]
            
    #writes image after inverse warping
    ###cv.imwrite("without holes.png",mapped_source_image)
   	    
    dest_image_height = dest_image.shape[0]
    dest_image_width = dest_image.shape[1]
	
    #get dimensions for the new image
    mosaic_image_height = max(mapped_source_height, dest_image_height+shiftHeight)
    mosaic_image_width= max(mapped_source_width, dest_image_width+shiftWidth)

    mosaic_image = np.zeros((mosaic_image_height, mosaic_image_width, 3), dtype=np.uint8);
    
    #put the second image in the mosaic image
    for i in range(0,dest_image_height):
        for j in range(0, dest_image_width):
            mosaic_image[i + shiftHeight][j + shiftWidth] = dest_image[i][j]

	#puts the first image in mosaic image
    for i in range(0,mapped_source_image.shape[0]):
        for j in range(0,mapped_source_image.shape[1]):
            if not( int(mapped_source_image[i][j][0]) == 0 and int(mapped_source_image[i][j][1]) == 0 and int(mapped_source_image[i][j][2]) == 0 ):
                mosaic_image[i][j] = mapped_source_image[i][j]
  
    
    return mosaic_image
    

def ransac_error(single_p,single_p_,h):
    "calcualtes difference between the calculated point and the given point"
    point = np.array([single_p[0],single_p[1],1])
    calculated_point_ = np.dot(h,point.reshape(-1,1))
    calculated_point_ /= calculated_point_[2]
    
    error = calculated_point_[0:2] - single_p_.reshape(-1,1) #difference
   
    error = np.square(error) #squared
    
    error = np.sum(error) #sum of dimensions
    
    return math.sqrt(error)

def ransac(p,p_,threshold,iterations):
    "takes coreespondance points and for each 4 random pairs it calculates h and counts inliners from a given threshold and keeps the best h"
    max_inliners = 0 
    best_h = None
   
    for i in range(iterations):
        inliners = 0 
        randp  = np.zeros([4,2]) 
        randp_ = np.zeros([4,2])
        H = None
        for j in range(4):
            random_index = random.randrange(0, p.shape[0],1) #picks random points from the given set
        
            randp[j]=p[random_index]
            randp_[j]=p_[random_index]
      
        H = calculate_h(randp,randp_) #calculates h from the 4 random points
        
       
        
        for j in range(p.shape[0]):
            error = ransac_error(p[j],p_[j],H)
           
            
            if(error<threshold): 
                inliners+=1
            
            
        if(inliners>max_inliners):
            max_inliners = inliners
            best_h = H
            #return the H and the number of inliners for H
    return best_h , max_inliners 
            
    
   
def construct_mosaic(path1,path2,correspondance_method,H_calculation_method,ransac_loops=100):
    image1 = cv.imread(path1)
    image2 = cv.imread(path2)
    
    image1_gray = cv.cvtColor(image1,cv.COLOR_RGB2GRAY)
    image2_gray = cv.cvtColor(image2,cv.COLOR_RGB2GRAY)
    
    new_filename = path1.split(".")[0]+"and"+path2.split(".")[0]
    
    corr_method_string = ""
    H_method_string = ""
    if(correspondance_method == CORRESPONDENCE_AUT0):
        p1,p2 = get_correspondance_auto(image1_gray,image2_gray)
        corr_method_string = "automatic correspondence"
    else:
        p1,p2 = get_correspondance_manually(image1,image2,8)
        corr_method_string = "manual correspondence"

    new_filename = new_filename +"_"+corr_method_string
       
    if(H_calculation_method==CALCULATE_H_NORMALLY):
        
        H = calculate_h(p1[0:90],p2[0:90])
        H_method_string = "H calcualated without ransac"
        
        
    else:
        H, inliners = ransac(p1,p2,5,ransac_loops)
        H_method_string = "H calcualated with ransac"
        new_filename = new_filename +"_"+H_method_string + "_" + "ransacloops="+str(ransac_loops)
    
    new_filename = new_filename +"_"+H_method_string 
    


    mosaic_image = warp(image1,image2,H)
    
    #writes the output
    cv.imwrite(new_filename+".png",mosaic_image)
    print("writing "+new_filename+" image is done")

 

#construct_mosaic("1.png","2.png",CORRESPONDENCE_AUT0,CALCULATE_H_RANSAC,100)
construct_mosaic("1.png","2.png",CORRESPONDENCE_MANUAL,CALCULATE_H_RANSAC,10)
#construct_mosaic("1.png","2.png",CORRESPONDENCE_AUT0,CALCULATE_H_RANSAC,1000)   
#construct_mosaic("1.png","2.png",CORRESPONDENCE_AUT0,CALCULATE_H_RANSAC,5000)
####
####
#construct_mosaic("1.png","2.png",CORRESPONDENCE_AUT0,CALCULATE_H_RANSAC,100)
#construct_mosaic("1.png","2.png",CORRESPONDENCE_AUT0,CALCULATE_H_NORMALLY)
###
###
#construct_mosaic("3.jpg","4.jpg",CORRESPONDENCE_AUT0,CALCULATE_H_RANSAC,1000)
#construct_mosaic("1.png","2.png",CORRESPONDENCE_AUT0,CALCULATE_H_NORMALLY)
