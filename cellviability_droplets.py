import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
##from skimage.transform import hough_ellipse
import imutils
import cv2
from glob import glob
import scipy.misc

#detect droplets-->
file_input='C:\\Users\melvin\Desktop\Codes\droplets\*.tif'
ind_img=glob(file_input)
for fn in ind_img:

    # load individual images from folder
    img=cv2.imread(fn)
    output_img = img.copy()

    # specify parameters for droplets to be detected using CHT (parameters can be changed depending on application)-replace this with FEHT for non-circular droplets 
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bimg = cv2.medianBlur(cimg, 5)
    #dimg = cv2.Canny(cimg, 5, 10)
    droplets = cv2.HoughCircles(bimg,cv2.HOUGH_GRADIENT,1.2,70,param1=100,param2=100,minRadius=50,maxRadius=100)
    if droplets is not None:
        droplets = np.round(droplets[0,:]).astype("int")
        for (x,y,r) in droplets:
            cv2.circle(output_img,(x,y),r,(0,255,0),2)
            cv2.rectangle(output_img, (x-5,y-5), (x+5,y+5), (0,125,255), -1)
##    # specify parameters for non-circular droplets using FEHT (parameters can be changed depending on application)-make sure to import the package
##    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##    bimg = cv2.medianBlur(cimg, 5)
##    #dimg = cv2.Canny(cimg, 5, 10)
##    droplets = hough_ellipse(edges, accuracy=20, threshold=250,min_size=50, max_size=150)
##    #parameterization
##    yc, xc, a, b = [int(round(x)) 
##    #Detect ellipse and draw outlines
##    my, mx = ellipse_perimeter(ym, xm, a, b, orientation)
##    output_img[cy, cx] = (0, 255, 255)
##    edges = color.gray2rgb(output_img(edges))
##    edges[cy, cx] = (255, 255, 255)       
    # print the number of detected droplets
        a=len(droplets)
        print (a)
        
##        v= np.array(output_img)
##        fig = plt.figure()
##        ax = fig.add_subplot(111, projection='3d')
##        ax.contour(v[:,0],v[:,1],v[:,2])
##        plt.show()
        
#detect live/dead/dying cells within droplets-->
##encapsulation = {} #use this for encapsulation
number = {}

file_input='C:\\Users\melvin\Desktop\Codes\droplets\*.tif'
ind_img=glob(file_input)
for fn in ind_img:
    img=cv2.imread(fn)
    height_orig, width_orig = img.shape[:2]
     
    # initialize output image
    contours_output = img.copy()

    # Detection of cells
    colors = ['live cell', 'dead cell','dying cell']
    for color in colors:
     
        # copy of original image
        image_to_process = img.copy()
     
        # initializes number to count the detected cells in loop
        #encapsulation[color]=0
        number[color] = 0
     
        # define NumPy arrays of color boundaries (BGR vectors)
        if color == 'live cell':
            lower = np.array([0, 153, 0])
            upper = np.array([112, 255, 112])
        elif color == 'dead cell':
            lower = np.array([ 0, 0, 153])
            upper = np.array([112, 112, 255])
        elif color == 'dying cell':
            lower = np.array([ 0, 153, 153])
            upper = np.array([112, 255, 255])
     
        # find the colors within the specified boundaries
        bound_img = cv2.inRange(image_to_process, lower, upper)
        # apply the mask
        resultant_img = cv2.bitwise_and(image_to_process, image_to_process, mask = bound_img)
     
        ## load the image, convert it to grayscale, and blur it slightly
        image_gray = cv2.cvtColor(resultant_img, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        cv2.imwrite('blur.tif',image_gray)
     
        # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
        image_edged = cv2.Canny(image_gray, 50, 100)
        cv2.imwrite('canny.tif',image_edged)
        image_dilate = cv2.dilate(image_edged, None, iterations=1)
        cv2.imwrite('dilate.tif',image_dilate)
        image_erode = cv2.erode(image_dilate, None, iterations=1)
        cv2.imwrite('erode.tif',image_erode)
     
        # find contours in the edge map
        numb = cv2.findContours(image_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        numb = numb[0] if imutils.is_cv2() else numb[1]
    
        # loop over the contours individually
        for c in numb:
             
            # if the contour is not sufficiently large, ignore it (change the paramter as per experiment)
            if cv2.contourArea(c) < 250:
                continue   
            # compute the Convex Hull of the contour
            hull = cv2.convexHull(c)
            if color == 'live cell':
                # prints contours in red color
                cv2.drawContours(contours_output,[hull],0,(0,0,255),2)
            elif color == 'dead cell':
                # prints contours in blue color
                cv2.drawContours(contours_output,[hull],0,(255,0,0),2)
            elif color == 'dying cell':
                # prints contours in purple color
                cv2.drawContours(contours_output,[hull],0,(119,23,94),2)
     
            number[color] += 1
     
        # Print the number of cells of each subtype
        print(("{} {} numbers".format(number[color],color)))
        
##        # Detect encapsulation
##        for c in numb:
##            C=cv2.moments(c)
##            cx=int(C["numb10"]/C["numb00"])
##            cy=int(C["numb01"]/C["numb00"])
##            def calculateOverlap(cx,cy)
##                for i in range(len(numb)):
##                    x,y,cx,cy=cv2.boundingCircle(numb[i])
##                    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+35, x:x+35], peaks8u[y:y+35, x:x+35])
##                    encapsulation[color] += 1
##                print (("{} {} encapsulation".format(encapsulation[color],color)))

# Writes the output image
cv2.imwrite('output.png',contours_output)
cv2.waitKey(0)
cv2.destroyAllWindows()



