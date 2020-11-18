import cv2
import numpy as np 
import math
from datetime import datetime

def imagePaddingFunc(image, window):
    #Image padding function to extend the outer edges of the image 
    #to the same values as the first few pixels along the edges

    #defining the height and width of image and filtering window 
    wW,  wH = window.shape[:2]
    
    #padding the  edges of the image to allow processing to 
    #start at the beginning of the image (needs to be int)  
    padding = ((wW -1) // 2)

    paddedImage = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    #initialising the output image matrix
    output = np.zeros((imH, imW), dtype="float32")
    
    return output, padding, paddedImage

def slideMean (output, paddedImage, typeWindow, padding, imH, imW, elem):
    #For sliding down Y in the image as all of X completes
    # this is applies for filters that use a genereric mean filterin style 
    
    for y in np.arange(padding, imH+padding): 
        #For sliding across X in the image
        for x in np.arange(padding, imW +padding):
            
            #getting the Region of Interest of an image by creating the window
            # to apply the filter to
            roi = paddedImage[y-padding:y+padding+1, x-padding:x+padding+1]

            #applying the element wise multiplication of the filter to the image
            weightWindow = (roi * typeWindow)

            #adding  all the elements up and dividing my nuber of elemets to find the mean
            mean = weightWindow.sum()/elem

            output[y - padding, x - padding] = mean

    output = np.array(output, dtype="uint8")
    return  output

def slideMedian(output, paddedImage, typeWindow, padding, imH, imW):
    #For sliding across X and down Y in the image for all pixels
    # this is applies for filters that use a genereric median filterin style 
    
    for y in np.arange(padding, imH+padding): 
        #For sliding across X in the image
        for x in np.arange(padding, imW +padding):
            #getting the Region of Interest of an image by creating the window
            # to apply the filter to
            roi = paddedImage[y-padding:y+padding+1, x-padding:x+padding+1]

            #Sort from lowest to highest value in the array 
            #Clipping and additional convolution is for the weighted median
            roiSorted = np.clip(np.sort(roi, axis=None) * typeWindow.flatten(),0,255)
            
            median = math.ceil(len(roiSorted)/2) - 1

            output[y - padding, x - padding] = roiSorted[median]
    output = np.array(output, dtype="uint8")
    return output

def histogramNorm(output, offset, padding, imH, imW):
    # rescale the output image to make sure the range of the image extends
    # to a gull greyscale range of [0, 255]

    oMax = np.amax(output)  #max value of o/p image
    oMin = np.amin(output)  #min value of o/p image
    for y in np.arange(padding, imH+padding): 
        for x in np.arange(padding, imW +padding):
            output[y-padding, x-padding] = (output[y-padding, x-padding]-oMin)*255/(oMax-oMin) + offset 
    return output

##Linear Functions##
def meanFilter(image, n):
    #creating a window of matrix nxn
    typeWindow = np.ones((n,n), dtype=np.int8)
   
    #creating a scalar to divide te overall mean window to bering intensity
    #back into the relevant range of 0 to 255
    elem = typeWindow.sum()
    
    #Image Padding
    output, padding, paddedImage = imagePaddingFunc(image, typeWindow)

    #apply slide mean function to find the mean of every ROI along the image
    output  = slideMean(output,paddedImage, typeWindow, padding, imH, imW, elem)

	# return the output image
    output = np.array(output, dtype="uint8")
    
    return output

def gaussianFilter(image, n, sigma):
    #similar to a mean filter but uses a gaussian mask where the centre values
    #have higher intensity the outer values

    typeWindow = np.ones((n,n), dtype=np.float)
    output, padding, paddedImage = imagePaddingFunc(image, typeWindow)
    midpos = math.ceil(n/2) - 1
    elem = 2 * np.pi * (sigma**2)

    #making the gaussian window
    for j in range (0,n):
        for i  in range (0,n):
            typeWindow[j][i] = (1/elem) * np.exp( -0.5*( (midpos-(i))**2 + (midpos-(j))**2)/(sigma**2))

    output  = slideMean(output, paddedImage, typeWindow ,padding, imH, imW, 1)
    
    #applying histogram normalisation, as sigma increases, overall image darkens
    output = histogramNorm(output, 0, padding, imH, imW)

	#return the output image
    return output

##Non Linear Functions##
def trimmedMean(image, n, trimVal):
    #creating a window (aka kernel)
    typeWindow = np.ones((n,n), dtype=np.int8)
    element = typeWindow.sum()

    output, padding, paddedImage = imagePaddingFunc(image, typeWindow)

    for y in np.arange(padding, imH+padding): 
        for x in np.arange(padding, imW +padding):
            roi = paddedImage[y-padding:y+padding+1, x-padding:x+padding+1]

            #applying the element wise multiplication of the filter to the image
            k = (roi * typeWindow)
            elem = element

            #sorting window (kernel) into a long single axis
            windowSorted = np.sort(k, axis=None)
            
            #Trimming the mean 
            for i in range (0,trimVal):
                windowSorted = np.delete(windowSorted,0,0)
                windowSorted = np.delete(windowSorted,len(windowSorted)-1,0)
                elem = elem-2

            #adding  all the elements up and dividing my nuber of elemets to find the mean
            mean = k.sum()/elem
            if mean < 0:
                mean = 0
            if mean > 255:
                mean  =  255

            output[y - padding, x - padding] = mean

	# return the output image
    output = np.array(output, dtype="uint8")
    return output

def medianFilter(image,n): 
    typeWindow = np.ones((n,n), dtype=np.int8)
    output, padding, paddedImage = imagePaddingFunc(image, typeWindow)
    output  = slideMedian(output,paddedImage, typeWindow, padding, imH, imW)

	# return the output image
    output = np.array(output, dtype="uint8")
    return output

def weightedMedianFilter(image,n, W): 
    typeWindow = np.ones((n,n), dtype=np.int8)
    typeWindow[math.floor(n/2)][math.floor(n/2)] = W
    output, padding, paddedImage = imagePaddingFunc(image, typeWindow)

    output  = slideMedian(output,paddedImage, typeWindow, padding, imH, imW)

    output = histogramNorm(output, 0, padding, imH, imW)

	# return the output image
    output = np.array(output, dtype="uint8")
    return output

def truncatedMedianFilter(image,n): 
    window = np.ones((n,n), dtype=np.int8)
    output, padding, paddedImage = imagePaddingFunc(image, window)

    #For sliding down Y in the image as all of X completes
    for y in np.arange(padding, imH+padding): 
        #For sliding across X in the image
        for x in np.arange(padding, imW +padding):
            
            #getting the Region of Interest of an image by creating the window
            # to apply the filter to
            roi = paddedImage[y-padding:y+padding+1, x-padding:x+padding+1]

            #Sort from lowest to highest value in the array 
            roiSorted = np.sort(roi, axis=None)
            median = math.ceil(len(roiSorted)/2)

            #Difference of first half and second half 
            upperDifference = roiSorted[len(roiSorted)-1]-roiSorted[median]
            lowerDifference = roiSorted[median]-roiSorted[0]

            i = n*n
            #Truncation Loop
            while i > 0:
                i = i-1     
                if upperDifference > lowerDifference: 
                    #delete highest value in the array
                    roiSorted = np.delete(roiSorted,len(roiSorted)-1, 0)     

                    #set new values
                    median = math.ceil(len(roiSorted)/2) 
                    upperDifference = roiSorted[len(roiSorted)-1]-roiSorted[median]
                    lowerDifference = roiSorted[median]-roiSorted[0]

                elif lowerDifference > upperDifference:
                    #delete lowest value in the array
                    roiSorted = np.delete(roiSorted,0, 0)

                    #set new values of median, upperDifference and lowerDifference
                    median = math.ceil(len(roiSorted)/2)
                    try:  #to avoid errors if there are 2 elements left  
                        upperDifference = roiSorted[len(roiSorted)-1]-roiSorted[median]
                    except:
                        median = 0 
                        break
            output[y - padding, x - padding] = roiSorted[median]
            
    # return the output image
    output = np.array(output, dtype="uint8")        
    return output

def adaptiveMedianFilter(image,n, W, c):
    #adaptiveMedianFilter(image, 3, 100, 10, 3)                

    #Setting the central position of the adaptive  weight
    midpos = math.ceil(n/2)-1
    #Central Weight
    W = W*np.ones((n,n), dtype=np.int8)
    
    #Setting d values  (distance from the centre of the mask)
    d = np.ones((n,n), dtype=np.float)
    for j in range (0,n):
            for i  in range (0,n):
                d[j][i] = np.sqrt(((midpos-(i))**2 + (midpos-(j))**2))

    #appling padding
    output, padding, paddedImage = imagePaddingFunc(image, W)

    for y in np.arange(padding, imH+padding): 
        #For sliding across X in the image
        for x in np.arange(padding, imW +padding):
            #getting the Region of Interest
            roi = paddedImage[y-padding:y+padding+1, x-padding:x+padding+1]
            
            #getting mean xbar and std deviation
            mean = roi.sum()/(n*n)
            if mean == 0: 
                output[y - padding, x - padding] = 0
            else:    
                std  = np.std(roi)

                #adaptive weight window
                weightWindow = np.array(W - c*d*std/mean, dtype= np.float)

                #if values are less than 0 set to zero
                weightWindow = weightWindow.clip(0)
                
                #flattening ROI and Wighting
                roiFlattened = roi.flatten()

                weightWindowFlattened = weightWindow.flatten()

                #finding the index of the lowest to highest value in the ROI
                roiSortedIndex = np.argsort(roiFlattened)[::]
                
                #flattening out the ROI
                roiSorted = np.sort(roiFlattened, axis=0)
                
                #arranging the Weight Window according to the index values
                weightWindowFlattenedSorted = np.zeros(len(weightWindowFlattened)-1, dtype=np.int8) #initialising
                for i in range (0,len(weightWindowFlattened)-1):
                    weightWindowFlattenedSorted[i] = weightWindowFlattened[roiSortedIndex[i]]

                #finding the sorted Weights median by (sum of the weights /2)rounded up 
                medianWeightWindow = math.ceil(weightWindowFlattenedSorted.sum()/2) 
                
                findAdaptiveMedian = int(0)  #initialising for adaptive median value
                indexAdaptiveMedian = int(0) #initialising for adaptive median value
                

                #finding the index of the adaptive weighted median value   
                while findAdaptiveMedian < medianWeightWindow:
                    findAdaptiveMedian =  findAdaptiveMedian + weightWindowFlattenedSorted[indexAdaptiveMedian]
                    indexAdaptiveMedian = indexAdaptiveMedian  +1
                
                #finding the adaptive weighted median    
                adaptiveWeightedMedian = roiSorted[indexAdaptiveMedian]

                output[y - padding, x - padding] = adaptiveWeightedMedian
    
    # return the output image
    output = np.array(output, dtype="uint8")        
    return output

def weightedRankFilter(img, n):
    typeWindow = np.ones((n,n), dtype=np.int8)
    output, padding, paddedImage = imagePaddingFunc(image, typeWindow)

    #For sliding down Y in the image as all of X completes
    for y in np.arange(padding, imH+padding): 
        #For sliding across X in the image
        for x in np.arange(padding, imW +padding):
            #getting the Region of Interest of an image by creating the window
            # to apply the filter to
            roi = paddedImage[y-padding:y+padding+1, x-padding:x+padding+1]

            #Sort from lowest to highest value in the array 
            roiSorted = np.sort(roi, axis=None)
            
            median = math.ceil(len(roiSorted)/2) - 1

            output[y - padding, x - padding] = roiSorted[median]

	# return the output image
    output = np.array(output, dtype="uint8")
    return output

if __name__ == "__main__":
    #Reading image files for image processing
    image = cv2.imread('foetus.png',  cv2.IMREAD_GRAYSCALE )
    imH, imW = image.shape[:2]

    #Canny Edge Detection Threshold Values
    threshold1 = 30
    threshold2 =  100


    ########### APPLYING THE FILTERS ###########

    ##--ORIGINAL IMAGE--##
    # cv2.imshow('original image', image) 
    # print('original image')
    # edgeOrig = cv2.Canny(image, threshold1, threshold2)
    # catOrig = cv2.hconcat([image,edgeOrig]) 
    # cv2.imshow('Original', catOrig)


    ##-- MEAN FILTER -- LINEAR --##
    # n = 3
    # meanFilter = meanFilter(image, n)
    # # cv2.imshow('mean image', meanFilter) 
    # print('mean complete')
    # edgeMean = cv2.Canny(meanFilter, threshold1, threshold2)
    # catMean = cv2.hconcat([meanFilter,edgeMean]) 
    # cv2.imshow('Mean, n =' + str(n), catMean)


    # #-- MEAN GAUSSIAN FILTER--LINEAR --## 
    # n  = 5
    # sigma = 10
    # # for sigma in range(5,9,1):
    # #     for n in range (7,10,2):
    # gaussFilter = gaussianFilter(image, n,sigma)
    # # cv2.imshow('filtered image - gaussian', gaussFilter) 
    # print('gaussian complete')
    # edgeGauss = cv2.Canny(gaussFilter, threshold1, threshold2)
    # catGauss = cv2.hconcat([gaussFilter,edgeGauss]) 
    # cv2.imshow('Gaussian Mean, n =' + str(n) +', sigma = '+ str(sigma), catGauss)


    ##-- MEDIAN FILTER -- NON LINEAR--##
    # n = 9
    # imageMed = medianFilter(image, n)
    # print('median complete')
    # cv2.imshow('median image', imageMed) 
    # edgeMed = cv2.Canny(imageMed, threshold1, threshold2)
    # # catMed = cv2.hconcat([imageMed,edgeMed]) 
    # # cv2.imshow('Median, n =' + str(n), catMed) 


    # # # # ##-- WEIGHTED MEDIAN FILTER -- NON LINEAR--##
    # n =11
    # W = 2
    # imageWmed = weightedMedianFilter(image, n, W)
    # print('weighted median complete')
    # # cv2.imshow('wighted median image', imageWmed) 
    # edgeWmed = cv2.Canny(imageWmed, threshold1, threshold2)
    # catWMed = cv2.hconcat([imageWmed,edgeWmed])
    # cv2.imshow('Wighted Median', catWMed) 


    # ##-- TRIMMED MEAN FILTER -- NON LINEAR--##
    # n =11
    # trimVal = 5
    # # for n in range (5,10,2):
    # for trimVal in range (3,8,2):
    #     imageTmean = trimmedMean(image, n, trimVal)
    #     print('trimmed mean complete')
    #     # cv2.imshow('trimmed mean image', trimmedMean) 
    #     edgeTmean = cv2.Canny(imageTmean, threshold1, threshold2)
    #     catTmean = cv2.hconcat([imageTmean,edgeTmean])
    #     cv2.imshow('Trimmed Mean, n =' + str(n) +', trimVal = '+ str(trimVal), catTmean) 


    # ##-- RANK MEDIAN FILTER -- NON LINEAR--##
    # n = 9
    # imageRmed = weightedRankFilter(image,n)
    # print('ranked median complete')
    # # cv2.imshow('wighted rank image', imageRmed) 
    # edgeRmed = cv2.Canny(imageRmed, threshold1, threshold2)
    # catTmed = cv2.hconcat([imageRmed,edgeRmed])
    # cv2.imshow('Ranked Median', catTmed) 
    

    # ##-- TRUNCATED MEDIAN FILTER -- NON LINEAR --##
    # n = 15
    # t0 = datetime.now()
    # truncMed = truncatedMedianFilter(image, n)
    # print('truncated median took '+str(datetime.now()-t0)+' to complete ' )
    # # cv2.imshow('truncatedMedianFilter image - Linear', truncatedMedianFilter) 
    # edgeTruncMed = cv2.Canny(truncMed, threshold1, threshold2)
    # catTruncMed = cv2.hconcat([truncMed,edgeTruncMed])
    # cv2.imshow('Truncated Median', catTruncMed) 


    ##-- ADAPTIVE MEDIAN FILTER -- NON LINEAR --##
    # n = 9
    # W = 100
    # c = 25
    # for n in range (9,12,2):
    #     for c in range (6,11,4):
    #         adapMed = adaptiveMedianFilter(image, n, W, c)
    #         print('adaptiveMedianFilter complete')
    #         # cv2.imshow('Truncated Median', catAdapMed) 
    #         edgeAdapMed = cv2.Canny(adapMed, threshold1, threshold2)
    #         catAdapMed = cv2.hconcat([adapMed,edgeAdapMed])
    #         cv2.imshow('Adaptive Median, n =' + str(n) +', c = '+ str(c), catAdapMed) 

    # print('All Windows Are Displayed')



    t0 = datetime.now()
    ##attempt at good filtering
    # truncMed = truncatedMedianFilter(image, 15)
    # print('trunc1 complete')
    # print(datetime.now()-t0)
    # t0 = datetime.now()
    # gaussFilter = gaussianFilter(truncMed, 5,6)
    # print('gauss1 complete')
    # cv2.imshow('trunc1 gauss1', gaussFilter)
    gaussFilter = cv2.imread('truncgauss.png',  cv2.IMREAD_GRAYSCALE )  
    imageWmed = weightedMedianFilter(gaussFilter, 3, 1.5)
    # gaussFilter = gaussianFilter(imageWmed, 5,6)
    cv2.imshow('weighted the gauss', imageWmed)
    print(datetime.now()-t0)
    t0 = datetime.now()
    gaussFilter = gaussianFilter(imageWmed, 5,6)
    truncMed = truncatedMedianFilter(imageWmed, 5) 
    print('trunc2 complete')
    print(datetime.now()-t0)
    # gaussFilter = gaussianFilter(gaussFilter, 3,6)
    print('gauss2 complete')
    cv2.imshow('final', gaussFilter)
    print(datetime.now()-t0)
    edgeGauss = cv2.Canny(gaussFilter, threshold1, threshold2)
    catGauss = cv2.hconcat([gaussFilter,edgeGauss]) 
    cv2.imshow('Foetus', catGauss)
    print(datetime.now()-t0)


    cv2.waitKey()

#################### END ####################
##ARCHIVES
    #Linear Filters:
    # mean done
    # Gaussian
    # trimmed mean confused 
    # Adaptive linear filters not tried 

    # meidan done
    # truncated median done 
    # morphological reconstruction
    # opening
    # closing

 #timer
    # t0 = datetime.now()
    # print(datetime.now()-t0)

    ##-- MORPHOLOGICAL 
    # kernel = np.ones((5,5),np.uint8)
    # erosion = cv2.erode(image,kernel,iterations = 1)
    # cv2.imshow('erosion', erosion) 

