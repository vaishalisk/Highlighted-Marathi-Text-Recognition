import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import cv2
import scipy.misc as misc
import matplotlib.pyplot as plt
import time
from skimage import io, filters
from numpy import linalg as LA

from imutils import perspective
from imutils import contours
import imutils
import argparse

from fuzzywuzzy import fuzz
import difflib
from nltk.util import ngrams
from nltk.corpus import udhr
from skimage.data import page
from skimage.filters import (threshold_sauvola)

def four_point_transform(image, pts):
	rect = order_points_old(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))


	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def order_points_old(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    
    cv2.imshow("image",frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        cv2.imwrite("image27.png",frame)
        cv2.imshow("image",frame)
        #frame=cv2.imread(name)
        #frame = cv2.resize(frame,(400,600))
        cv2.imwrite("resized.png",frame)
        img=Image.open("resized.png")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # fluroscent pink
        '''
        lower_bound = np.array([151,100,100])        
        upper_bound = np.array([171,255,255])'''
        # fluroscent yellow
        lower_bound = np.array([20,100,100])        
        upper_bound = np.array([40,255,255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        median = cv2.medianBlur(res,15)
        cv2.imwrite("median.png",median)
        grayscaled = cv2.cvtColor(median,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grayscaled,10,255,cv2.THRESH_BINARY)
        edged = cv2.Canny(thresh, 30, 150)
        #cv2.imwrite('mask.png',mask)
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        frame_copy=frame
        cv2.drawContours(frame_copy, cnts[0], -1, (0,255,0), 3)
        cv2.imshow("input",frame_copy)
        BlackPixelMatrix=cnts[0]
        '''minCol=cnts[0][0][0][0]
        maxCol=cnts[0][0][0][0]
        minRow=cnts[0][0][0][1]
        maxRow=cnts[0][0][0][1]

        for j in range(0,len(cnts[0])):
                    if(minCol>(cnts[0][j][0][0])):
                        minCol=cnts[0][j][0][0]
                    if(maxCol<(cnts[0][j][0][0])):
                        maxCol=cnts[0][j][0][0]
                    if(minRow>(cnts[0][j][0][1])):
                        minRow=cnts[0][j][0][1]
                    if(maxRow<(cnts[0][j][0][1])):
                        maxRow=cnts[0][j][0][1]
            
            #print("minCol : {} ,maxCol : {} , minRow : {} , maxRow : {} ".format(minCol,maxCol,minRow,maxRow))
        mask1=Image.open("resized.png")
        newmask=mask1.crop((minCol-20,minRow-20,maxCol+20,maxRow+20))
        newmask=newmask.convert("RGB")
        newmask.save("masked.png")
        mask2=cv2.imread("masked.png")
        newmask=mask2
        col,row,_=newmask.shape
        print(newmask.shape)
        print("col : {} ,row : {} " .format(col,row))
        for rw in range(0,row-1):
                for c in range(0,col-1):
                    b=newmask[c][rw][0]
                    g=newmask[c][rw][1]
                    r=newmask[c][rw][2]
                    if(b<12):
                        if(g<150):
                            if(r<125):
                                #print("pixel value : {} {} {}" .format(b,g,r))
                                newmask[c][rw]=[0,0,0]
                            else:
                                newmask[c][rw]=[255,255,255]
                        else:
                            newmask[c][rw]=[255,255,255]
                    else:
                        newmask[c][rw]=[255,255,255]

        cv2.imshow("changed image",newmask)
        white_bgd_img=newmask

            
        BlackPixelMatrix=[]
            #trial_image=cv2.imread("trial_image.png")
            #(row,col,_)=trial_image.shape
        (row,col,_)=white_bgd_img.shape
        for x in range (0,row-1):
                    for z in range(0,col-1):
                        if (np.all((white_bgd_img[x,z]==[0,0,0]))):
                            BlackPixelMatrix.append([x,z])'''

        BlackPixelMatrix=np.array(BlackPixelMatrix)
        x=[]
        y=[]
        for h in range (len(BlackPixelMatrix)-1):
                x.append(BlackPixelMatrix[h][0][0])
                y.append(BlackPixelMatrix[h][0][1])
        x = x - np.mean(x)
        y = y - np.mean(y)
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        evec1, evec2 = evecs[:, sort_indices]
        x_v1, y_v1 = evec1  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evec2
        theta = np.rad2deg((np.arctan((y_v1)/(x_v1))))
        theta2 = np.rad2deg((np.arctan((y_v2)/(x_v2))))
        print("theta : {} , {}" .format(theta,theta2))
        RotateImg1 = img.rotate((0-theta))
        RotateImg1.save("RotatedImage"+".jpg")
        rotated=cv2.imread("RotatedImage.jpg")
        cv2.imshow("Rotated Image",rotated)
        hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(rotated,rotated, mask= mask)
        median3 = cv2.medianBlur(res,15)
        grayscaled = cv2.cvtColor(median3,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grayscaled,10,255,cv2.THRESH_BINARY)
        edged = cv2.Canny(thresh, 30, 150)
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        frame_copy=frame
        cv2.drawContours(frame_copy, cnts[0], -1, (0,255,0), 3)
        cv2.imshow("input",frame_copy)
        BlackPixelMatrix2=cnts[0]
        BlackPixelMatrix2=np.array(BlackPixelMatrix2)
        x2=[]
        y2=[]
        for h in range (len(BlackPixelMatrix2)-1):
                x2.append(BlackPixelMatrix2[h][0][0])
                y2.append(BlackPixelMatrix2[h][0][1])
        x2 = x2 - np.mean(x)
        y2 = y2 - np.mean(y)
        coords2 = np.vstack([x2, y2])
        cov2 = np.cov(coords2)
        evals2, evecs2 = np.linalg.eig(cov2)
        sort_indices2 = np.argsort(evals2)[::-1]
        evec1_2, evec2_2 = evecs2[:, sort_indices2]
        x_v1, y_v1 = evec1_2  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evec2_2
        theta_2 = np.rad2deg((np.arctan((y_v1)/(x_v1))))
        theta2_2 = np.rad2deg((np.arctan((y_v2)/(x_v2))))
        print("theta2 : {} , {}" .format(theta_2,theta2_2))
        if(abs(theta_2)>5):
                RotateImg2 = img.rotate((0-theta2))
                RotateImg2.save("RotatedImage"+".jpg")
                rotated=cv2.imread("RotatedImage.jpg")
                cv2.imshow("Rotated Image2",rotated)
                        
        ###########TILT################
        frame=rotated
        ap = argparse.ArgumentParser()
        ap.add_argument("-n", "--new", type=int, default=-1,
                help="whether or not the new order points should should be used")
        args = vars(ap.parse_args())
        hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(rotated,rotated, mask= mask)
        median3 = cv2.medianBlur(res,15)
        grayscaled = cv2.cvtColor(median3,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grayscaled,10,255,cv2.THRESH_BINARY)
        edged = cv2.Canny(thresh, 30, 150)
        edged_cp=edged
        # find contours in the edge map
        (_,cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        #print(cnts)
        cv2.imshow("cnt1",edged_cp)
        '''newCnts=cnts
        cnt_len_list=[]
        max_len_index=0        
        if(len(newCnts)>2):
                for h in range(0,(len(newCnts)-2)):
                    
                    if((len(newCnts[max_len_index]))<(len(newCnts[h+1]))):
                        max_len_index=h+1;
                
        else:
                if(len(newCnts)==2):
                    if(len(newCnts[0])<len(newCnts[1])):
                        max_len_index=1
                    else:
                        max_len_index=0
        print("max_len_index :" .format(max_len_index))
        cv2.drawContours(mask, cnts[max_len_index], -1, (0,255,0), 3)
        cv2.imshow("cnt",edged_cp)'''
        # sort the contours from left-to-right and initialize the bounding box
        # point colors
        
        (cnts, _) = contours.sort_contours(cnts)
        #cnts=cnts[max_len_index]
        colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
        # loop over the contours individually
        for (i, c) in enumerate(cnts):
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < 100:
                        continue

                # compute the rotated bounding box of the contour, then
                # draw the contours
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

                # show the original coordinates
                print("Object #{}:".format(i + 1))
                print('box =',box)

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                rect = order_points_old(box)

                # check to see if the new method should be used for
                # ordering the coordinates
                if args["new"] > 0:
                        rect = perspective.order_points(box)

                # show the re-ordered coordinates
                print(rect.astype("int"))
                print("")

                # loop over the original points and draw them
                #for ((x, y), color) in zip(rect, colors):
                        #cv2.circle(frame, (int(x), int(y)), 5, color, -1)

                # draw the object num at the top-left corner
                cv2.putText(frame, "Object #{}".format(i + 1),
                        (int(rect[0][0] - 15), int(rect[0][1] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # show the image
                cv2.imshow("Image", frame)
                #plt.imshow(frame)
                #plt.show()
                print("box:" .format(box))
        
        warped = four_point_transform(frame, box)
        cv2.imshow("Original", frame)
        ################################ Illumination ########################
        cv2.imshow("Warped", warped)
        cv2.imwrite("Warped.png", warped)
        (a,b,c)=warped.shape
        print(a)
        print(b)
        print(c)
        image = cv2.imread("Warped.png",0)
        window_size = 31
        thresh_sauvola = threshold_sauvola(image, window_size=window_size)
        binary_sauvola = image > thresh_sauvola
        plt.imshow(binary_sauvola)
        plt.axis('off')
        plt.savefig('SauvolaThreshold.png')
        image=cv2.imread('SauvolaThreshold.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        text = pytesseract.image_to_string(Image.open(filename),lang='mar',config='--psm 8')
        #text = pytesseract.image_to_string(Image.open(filename),lang='eng',config='--psm 8')
        os.remove(filename)
        print("Detected Word : " + text)
        
        f = open('raw.txt',encoding="utf8")
        message = f.read()
        term=text
        def get_best_match(term, message):
            ngs = ngrams( list(message), len(term) )
            ngrams_text = [''.join(x) for x in ngs]
            return difflib.get_close_matches(term, ngrams_text, n=1, cutoff=0)
        if term in message:
            print('Present')
        else:
            print('Absent')
        match = get_best_match(term,message)
        print(match)
        list1=(message.split())
        #print(list1)
        print(difflib.get_close_matches(term,list1))
        f.close()
            
        break    
    
cap.release()


