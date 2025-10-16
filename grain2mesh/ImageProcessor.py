"""
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Class definition for ImageProcessor

Functionality:
- Load base image
- Filter image to remove noise
- Segment image using watershed algorithm

"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage import morphology, measure, segmentation
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import copy



class ImageProcessor:
    def __init__(self):
        pass

    def load_image(self, image_basename):
        image_filename = os.path.join(os.getcwd(), image_basename)
        data = np.load(image_filename)
        image = data['array1']
        image_name = os.path.splitext(os.path.basename(image_basename))[0]

        return image, image_name

    def apply_gaussian(self, image, sigma):
        filtered = gaussian_filter(image.astype(float), sigma=sigma) # Gaussian smoothing
        binary_filtered_image = (filtered > 0.5).astype(np.uint8)    # binary mask

        return binary_filtered_image

    # OTHER FILTERS
    # def closing(self, image, radius):
    #     return morphology.closing(image, morphology.disk(radius))

    # def opening(self, image, radius):
    #     return morphology.opening(image, morphology.disk(radius))

    # def morphological_gradient(self, image, radius):
    #     return morphology.morphological_gradient(image, morphology.disk(radius))

    def watershed(self, image, sigma, min_dist, threshold):
        """
        Performs watershed segmentation on an image array

        Args:
        - image (np.ndarray): input image
        - sigma (float): blur amount for distance map
        - min_dist (int): minimum distance separating peaks
        - threshold (float): minimum intensity of peaks

        Returns:
        - labels (np.ndarray): image array labled by segmented region
        """
        # Distance transform and smoothing
        distance = distance_transform_edt(image)
        smooth_distance = gaussian(distance, sigma=sigma)

        # Find local maxima
        local_maxi = peak_local_max(smooth_distance, min_distance=min_dist, threshold_abs=threshold)
        local_maxi_binary = np.zeros_like(smooth_distance, dtype=bool)
        local_maxi_binary[tuple(local_maxi.T)] = True

        # Label markers
        markers = measure.label(local_maxi_binary)

        # Perform watershed segmentation
        labels = segmentation.watershed(-smooth_distance, markers, mask=image)
        return labels

    def watershed_filtering(self,image,filtered_image,Params):
        """
        Contribued by beta tester Evan Bozek
        Performs watershed segmentation on an unfiltered image and applies, takes the material structure after gaussian filtering, and reassigns new material to already assigned grain segments
        (keeps porosity from filtered image but preserves grain structure from original image)

        Args:
        - image (np.ndarray): input image
        filtered_image (np.ndarray): gaussian filtered image
        - Params class: grain2mesh class structure with filtering parameters
        
        
        Returns:
        - segmented image3 (np.ndarray): segmented image 
        """
        # retrieve image size
        n=np.size(image,0)
        m=np.size(image,1)
        # segment the unfiltered image
        segmented_image2 = self.watershed(image, Params.watershed_sigma, Params.peak_min_distance, Params.watershed_peak_threshold)
        segmented_image2=segmented_image2*filtered_image
        Properties=['area','coords']
        # make sure  regions have not been disconnected
        toAddx=np.zeros(0)
        toAddy=np.zeros(0)
        M=np.max(segmented_image2).astype(int)
        for I in range(1,M+1):
            TEMPIM=segmented_image2==I
            TEMPIMlabeled=measure.label(TEMPIM)
            TABLE=measure.regionprops_table(TEMPIMlabeled,TEMPIM,properties=Properties)
            L=len(TABLE['area'])
            if L>1:
                IND=np.argmax(TABLE['area'])
                INDS=list(np.linspace(0,L-1,L).astype(int))
                INDS.remove(IND)
                for II in INDS:
                    x=TABLE['coords'][II][:,0]
                    y=TABLE['coords'][II][:,1]
                    toAddx=np.append(toAddx,np.array(x))
                    toAddy=np.append(toAddy,np.array(y))
                    #for III in len(x):
                    segmented_image2[x,y]=0
                                    
                
            
            
        
        
        #create copy of segmented image
        segmented_image3=copy.deepcopy(segmented_image2)
        
        # find pixels that were changed from porosity to material due to filtering
        im=(((segmented_image2)>=1)^(filtered_image==1) ) &(filtered_image==1)
        
        #get locationd of changed pixels
        PIX=np.where(im)
        PIX_X=PIX[0]
        PIX_Y=PIX[1]
        # add disconnected pixels to list of pixels to resegment and make sure they are unique
        TEMP2=np.array([toAddx,toAddy]).transpose()
        TEMP=np.append(np.expand_dims(PIX_X, 1),np.expand_dims(PIX_Y, 1),axis=1)

        TEMP=np.append(TEMP,TEMP2,axis=0)
        TEMP=list(TEMP)
        for I,val in enumerate(TEMP):
            TEMP[I]=tuple(val)
        TEMP=np.array(list(set(TEMP)))
        if len(TEMP)>0:
            PIX_X=TEMP[:,0].astype(int)
            PIX_Y=TEMP[:,1].astype(int)
            
            L=len(PIX_X)
            
            skipped=[] # index value (within PIX) of skipped pixels
            for I in range(L):# loop through each pixel that was changed from prorsity to material
                x=PIX_X[I]
                y=PIX_Y[I]
                
                #get adjacent coordinates to search
                COORDS=np.array([x+1,x-1,y-1,y+1])
                
                # Make sure coords stay within image bounds
                COORDS[COORDS<0]=0
                if (x+1)>=n:
                    COORDS[0]=n-1
                if (y+1)>=m:
                    COORDS[3]=m-1
                # list of adjacent segment labels: [up,down,right,left]
                TEMP=[segmented_image2[x,COORDS[3]] ,segmented_image2[x,COORDS[2]],segmented_image2[COORDS[0],y],segmented_image2[COORDS[1],y] ]
                
                # discount porosity (0) labels from consideration
                if TEMP.count(int(0))>0:
                    TEMP=list(filter(lambda a: a!=0,TEMP))
                    
                if len(TEMP)==0: # no nonzero nearby segments
                    skipped.append(I) # re-analyze later
                    label=0
                    
                else:
                    
                    # find the unique adjacent segment labels, and the respective counts
                    [LABELS,COUNTS]=np.unique(TEMP,return_counts=True) 
                    
                    if len(LABELS)>1: # if there is more than one nearby grain
                        [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True) # find if number of counts is repeated
                        
                        if len(LABELS2)==1: #if all counts are equal (one of two/three segments, or two of two segments)
                            skipped.append(I)# 
                            label=0
                            
                        else:# one label was found more than the others
                            M=np.argmax(COUNTS)
                            label=LABELS[M]
                           
                    else: #only one nearby grain segment 
                        label=LABELS[0]
                #assign found label to this pixel, or zero if skipped
                segmented_image3[x,y]=label
                            
            # redo analysis on grains that were tied in nearby segment labels, or completely surrounded by pore space     
            # works to "erode" the unfound pixel labels
            
            L1=len(skipped)
            L2=0
            while L2!=L1: # while the number of skipped grains is chainging. all that will be left are ties
            
                skipped2=copy.deepcopy(skipped)
                skipped=[]
                for I in range(L1):
                    
                    #get x and y coord of pixel to assign a new label
                    x=PIX_X[skipped2[I]]
                    y=PIX_Y[skipped2[I]]
                    
                    #get adjacent coordinates to search
                    COORDS=np.array([x+1,x-1,y-1,y+1])
                    
                    # Make sure coords stay within image bounds
                    COORDS[COORDS<0]=0
                    if (x+1)>=n:
                        COORDS[0]=n-1
                    if (y+1)>=m:
                        COORDS[3]=m-1
                        
                    # list of adjacent segment labels : [up,down,right,left]
                    TEMP=[segmented_image3[x,COORDS[3]] ,segmented_image3[x,COORDS[2]],segmented_image3[COORDS[0],y],segmented_image3[COORDS[1],y] ]
                    #discount porosity labels from consideration
                    if TEMP.count(int(0))>0:
                        TEMP=list(filter(lambda a: a!=0,TEMP))
                        
                    if len(TEMP)==0: # no nonzero
                        # re-analyze later
                        skipped.append(skipped2[I])
                        label=0
                        
                    else:
                        
                        # find the unique adjacent segment labels, and the respective counts
                        [LABELS,COUNTS]=np.unique(TEMP,return_counts=True)
                        
                        if len(LABELS)>1: # if there is more than one nearby grain
                            [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True)# find if number of counts is repeated
                            
                            if len(LABELS2)==1: #if all counts are equal
                            #re-analyze later
                                skipped.append(skipped2[I])
                                label=0
                              
                            else:# one label was found more than the others
                            #set new label for this pixel
                                M=np.argmax(COUNTS)
                                label=LABELS[M]
                               
                        else: #only one nearby grain
                            label=LABELS[0]
                    #assign found label to this pixel, or zero if skipped        
                    segmented_image3[x,y]=label
                L2=L1 # how many pixels were skipped the previous iteration
                L1=len(skipped) # how many pixels were skipped this iteration
            segmented_image2=copy.deepcopy(segmented_image3)
            
            #reset while loop condition
            L1=len(skipped)
            L2=0
            while L2!=L1: # while the number of skipped grains is chainging. all that will be left are ties
                skipped2=copy.deepcopy(skipped)
                skipped=[]
                
                for I in range(L1):
                    #get x and y coord of pixel to assign a new label
                    x=PIX_X[skipped2[I]]
                    y=PIX_Y[skipped2[I]]
                    
                    #get adjacent coordinates to search
                    COORDS=np.array([x+1,x-1,y-1,y+1])
                    
                    # Make sure coords stay within image bounds
                    COORDS[COORDS<0]=0
                    if (x+1)>=n:
                        COORDS[0]=n-1
                    if (y+1)>=m:
                        COORDS[3]=m-1
                        
                    # list of 8 surrounding segment labels  
                    TEMP=[segmented_image3[x,COORDS[3]] ,segmented_image3[x,COORDS[2]],segmented_image3[COORDS[0],y],segmented_image3[COORDS[1],y],segmented_image3[COORDS[1],COORDS[2]], segmented_image3[COORDS[1],COORDS[3]], segmented_image3[COORDS[0],COORDS[2]], segmented_image3[COORDS[0],COORDS[3]]  ]
                    
                    # discount porosity (0) labels from consideration
                    if TEMP.count(int(0))>0:
                        TEMP=list(filter(lambda a: a!=0,TEMP))
                        
                    if len(TEMP)==0: # if no nonzero segments nearby
                    # re-analyze later (should not hppen at this point)
                        skipped.append(skipped2[I])
                        label=0
                        
                    else:
                        # find the unique adjacent segment labels, and the respective counts
                        [LABELS,COUNTS]=np.unique(TEMP,return_counts=True)
                        
                        if len(LABELS)>1: # if there is more than one nearby grain segment label
                        #find if number of counts is repeated
                            [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True) 
                            
                            if len(LABELS2)==1: #if all counts are equal
                            # re-analyze later
                                skipped.append(skipped2[I])
                                label=0
                                
                            else: # one label was found more than the others                        
                                #set new label for pixel
                                M=np.argmax(COUNTS)
                                label=LABELS[M]
                                #segmented_image3[x,y]=LABELS[M]
                        else: #only one nearby grain
                            label=LABELS[0]
                    #assign found label to this pixel, or zero if skipped
                    segmented_image3[x,y]=label
                
                # redo analysis on grains that were tied in nearby segment labels, or completely surrounded by pore space     
                # works to "erode" the unfound pixel labels
                L2=L1
                L1=len(skipped)
                
            #segmented_image2=copy.deepcopy(segmented_image3)
            
            
            
            #force assignment of ties
            for I in range(L1): # loop thorugh remaining skipped pixel assignments
                #get x and y coord of pixel to assign a new label
                x=PIX_X[skipped2[I]]
                y=PIX_Y[skipped2[I]]
                
                #get adjacent coordinates to search
                COORDS=np.array([x+1,x-1,y-1,y+1])
                
                # Make sure coords stay within image bounds
                COORDS[COORDS<0]=0
                if (x+1)>=n:
                    COORDS[0]=n-1
                if (y+1)>=m:
                    COORDS[3]=m-1
                    
                # list of 8 surrounding segment labels    
                TEMP=[segmented_image3[x,COORDS[3]] ,segmented_image3[x,COORDS[2]],segmented_image3[COORDS[0],y],segmented_image3[COORDS[1],y],segmented_image3[COORDS[1],COORDS[2]], segmented_image3[COORDS[1],COORDS[3]], segmented_image3[COORDS[0],COORDS[2]], segmented_image3[COORDS[0],COORDS[3]]  ]
                
                # discount porosity (0) labels from consideration
                if TEMP.count(int(0))>0:
                    TEMP=list(filter(lambda a: a!=0,TEMP))
                # find the unique adjacent segment labels, and the respective counts
                [LABELS,COUNTS]=np.unique(TEMP,return_counts=True)
                #force assign first segment found (coupd probabply find better condition here later if desired)
                try:
                    label=LABELS[0]
                except:# rare case where floating piece is left
                    label=0
                segmented_image3[x,y]=label
            
            segmented_image3=segmented_image3*filtered_image
        return segmented_image3
    def check_border(self,segmented_image2):
        m=np.size(segmented_image2,0)
        n=np.size(segmented_image2,1)
        border_sum=np.sum([np.sum(segmented_image2[0:,0]),np.sum(segmented_image2[0:,-1]),np.sum(segmented_image2[0,0:]),np.sum(segmented_image2[-1,0:])])
        
        if border_sum==0:
            segmented_image3=copy.deepcopy(segmented_image2)
            #ass borderindicies to list without repeating corneres
            I_X=np.linspace(0,m-1,m).astype(int)
            I_Y=np.linspace(1,n-2,n-2).astype(int)
            zeros_X=np.zeros(m).astype(int)
            zeros_Y=np.zeros(n-2).astype(int)
            PIX_X=np.array(I_X)
            PIX_X=np.append(PIX_X,I_X)
            PIX_Y=np.array(zeros_X)
            PIX_Y=np.append(PIX_Y,(n-1)*(zeros_X+1))           
            PIX_X=np.append(PIX_X,zeros_Y)
            PIX_Y=np.append(PIX_Y,I_Y)
            PIX_X=np.append(PIX_X,(m-1)*(zeros_Y+1)).astype(int)
            PIX_Y=np.append(PIX_Y,I_Y).astype(int)
            
            L=len(PIX_X)
            skipped=[]
            for I in range(L):# loop through each pixel that was changed from prorsity to material
                x=PIX_X[I]
                y=PIX_Y[I]
                
                #get adjacent coordinates to search
                COORDS=np.array([x+1,x-1,y-1,y+1])
                
                # Make sure coords stay within image bounds
                COORDS[COORDS<0]=0
                if (x+1)>=n:
                    COORDS[0]=n-1
                if (y+1)>=m:
                    COORDS[3]=m-1
                # list of adjacent segment labels: [up,down,right,left]
                TEMP=[segmented_image2[x,COORDS[3]] ,segmented_image2[x,COORDS[2]],segmented_image2[COORDS[0],y],segmented_image2[COORDS[1],y] ]
                
                # discount porosity (0) labels from consideration
                if TEMP.count(int(0))>0:
                    TEMP=list(filter(lambda a: a!=0,TEMP))
                    
                if len(TEMP)==0: # no nonzero nearby segments
                    #skipped.append(I) # re-analyze later
                    label=0
                    
                else:
                    
                    # find the unique adjacent segment labels, and the respective counts
                    [LABELS,COUNTS]=np.unique(TEMP,return_counts=True) 
                    
                    if len(LABELS)>1: # if there is more than one nearby grain
                        [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True) # find if number of counts is repeated
                        
                        if len(LABELS2)==1: #if all counts are equal (one of two/three segments, or two of two segments)
                            skipped.append(I)# 
                            label=0
                            
                        else:# one label was found more than the others
                            M=np.argmax(COUNTS)
                            label=LABELS[M]
                           
                    else: #only one nearby grain segment 
                        label=LABELS[0]
                #assign found label to this pixel, or zero if skipped
                segmented_image3[x,y]=label
                            
            # redo analysis on grains that were tied in nearby segment labels, or completely surrounded by pore space     
            # works to "erode" the unfound pixel labels
            
            L1=len(skipped)
            L2=0
            while L2!=L1: # while the number of skipped grains is chainging. all that will be left are ties
            
                skipped2=copy.deepcopy(skipped)
                skipped=[]
                for I in range(L1):
                    
                    #get x and y coord of pixel to assign a new label
                    x=PIX_X[skipped2[I]]
                    y=PIX_Y[skipped2[I]]
                    
                    #get adjacent coordinates to search
                    COORDS=np.array([x+1,x-1,y-1,y+1])
                    
                    # Make sure coords stay within image bounds
                    COORDS[COORDS<0]=0
                    if (x+1)>=n:
                        COORDS[0]=n-1
                    if (y+1)>=m:
                        COORDS[3]=m-1
                        
                    # list of adjacent segment labels : [up,down,right,left]
                    TEMP=[segmented_image3[x,COORDS[3]] ,segmented_image3[x,COORDS[2]],segmented_image3[COORDS[0],y],segmented_image3[COORDS[1],y] ]
                    #discount porosity labels from consideration
                    if TEMP.count(int(0))>0:
                        TEMP=list(filter(lambda a: a!=0,TEMP))
                        
                    if len(TEMP)==0: # no nonzero
                        # re-analyze later
                        skipped.append(skipped2[I])
                        label=0
                        
                    else:
                        
                        # find the unique adjacent segment labels, and the respective counts
                        [LABELS,COUNTS]=np.unique(TEMP,return_counts=True)
                        
                        if len(LABELS)>1: # if there is more than one nearby grain
                            [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True)# find if number of counts is repeated
                            
                            if len(LABELS2)==1: #if all counts are equal
                            #re-analyze later
                                skipped.append(skipped2[I])
                                label=0
                              
                            else:# one label was found more than the others
                            #set new label for this pixel
                                M=np.argmax(COUNTS)
                                label=LABELS[M]
                               
                        else: #only one nearby grain
                            label=LABELS[0]
                    #assign found label to this pixel, or zero if skipped        
                    segmented_image3[x,y]=label
                L2=L1 # how many pixels were skipped the previous iteration
                L1=len(skipped) # how many pixels were skipped this iteration
            segmented_image2=copy.deepcopy(segmented_image3)
            
            #reset while loop condition
            L1=len(skipped)
            L2=0
            while L2!=L1: # while the number of skipped grains is chainging. all that will be left are ties
                skipped2=copy.deepcopy(skipped)
                skipped=[]
                
                for I in range(L1):
                    #get x and y coord of pixel to assign a new label
                    x=PIX_X[skipped2[I]]
                    y=PIX_Y[skipped2[I]]
                    
                    #get adjacent coordinates to search
                    COORDS=np.array([x+1,x-1,y-1,y+1])
                    
                    # Make sure coords stay within image bounds
                    COORDS[COORDS<0]=0
                    if (x+1)>=n:
                        COORDS[0]=n-1
                    if (y+1)>=m:
                        COORDS[3]=m-1
                        
                    # list of 8 surrounding segment labels  
                    TEMP=[segmented_image3[x,COORDS[3]] ,segmented_image3[x,COORDS[2]],segmented_image3[COORDS[0],y],segmented_image3[COORDS[1],y],segmented_image3[COORDS[1],COORDS[2]], segmented_image3[COORDS[1],COORDS[3]], segmented_image3[COORDS[0],COORDS[2]], segmented_image3[COORDS[0],COORDS[3]]  ]
                    
                    # discount porosity (0) labels from consideration
                    if TEMP.count(int(0))>0:
                        TEMP=list(filter(lambda a: a!=0,TEMP))
                        
                    if len(TEMP)==0: # if no nonzero segments nearby
                    # re-analyze later (should not hppen at this point)
                        skipped.append(skipped2[I])
                        label=0
                        
                    else:
                        # find the unique adjacent segment labels, and the respective counts
                        [LABELS,COUNTS]=np.unique(TEMP,return_counts=True)
                        
                        if len(LABELS)>1: # if there is more than one nearby grain segment label
                        #find if number of counts is repeated
                            [LABELS2,COUNTS2]=np.unique(COUNTS,return_counts=True) 
                            
                            if len(LABELS2)==1: #if all counts are equal
                            # re-analyze later
                                skipped.append(skipped2[I])
                                label=0
                                
                            else: # one label was found more than the others                        
                                #set new label for pixel
                                M=np.argmax(COUNTS)
                                label=LABELS[M]
                                #segmented_image3[x,y]=LABELS[M]
                        else: #only one nearby grain
                            label=LABELS[0]
                    #assign found label to this pixel, or zero if skipped
                    segmented_image3[x,y]=label
                
                # redo analysis on grains that were tied in nearby segment labels, or completely surrounded by pore space     
                # works to "erode" the unfound pixel labels
                L2=L1
                L1=len(skipped)
                
            #segmented_image2=copy.deepcopy(segmented_image3)
            
            
            
            #force assignment of ties
            for I in range(L1): # loop thorugh remaining skipped pixel assignments
                #get x and y coord of pixel to assign a new label
                x=PIX_X[skipped2[I]]
                y=PIX_Y[skipped2[I]]
                
                #get adjacent coordinates to search
                COORDS=np.array([x+1,x-1,y-1,y+1])
                
                # Make sure coords stay within image bounds
                COORDS[COORDS<0]=0
                if (x+1)>=n:
                    COORDS[0]=n-1
                if (y+1)>=m:
                    COORDS[3]=m-1
                    
                # list of 8 surrounding segment labels    
                TEMP=[segmented_image3[x,COORDS[3]] ,segmented_image3[x,COORDS[2]],segmented_image3[COORDS[0],y],segmented_image3[COORDS[1],y],segmented_image3[COORDS[1],COORDS[2]], segmented_image3[COORDS[1],COORDS[3]], segmented_image3[COORDS[0],COORDS[2]], segmented_image3[COORDS[0],COORDS[3]]  ]
                
                # discount porosity (0) labels from consideration
                if TEMP.count(int(0))>0:
                    TEMP=list(filter(lambda a: a!=0,TEMP))
                # find the unique adjacent segment labels, and the respective counts
                [LABELS,COUNTS]=np.unique(TEMP,return_counts=True)
                #force assign first segment found (coupd probabply find better condition here later if desired)
                try:
                    label=LABELS[0]
                except:# rare case where floating piece is left
                    label=0
                segmented_image3[x,y]=label
            
            
            
            
            
            
        else:
                
            segmented_image3=segmented_image2
        return segmented_image3