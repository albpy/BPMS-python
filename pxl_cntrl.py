import cv2 as cv 
from utils import *
import datetime
import os
import numpy as np

class Frame:
    def __init__(self, frame=None):
        self.frame = frame
    def save_frame(self, frame):
        now=datetime.datetime.now()
        filename = "frames/frame_{}.png".format(now.strftime("%Y-%m-%d_%H-%M-%S"))
        cv.imwrite(filename, frame)


class Image:
    def __init__(self, Img = None):
        self.img = Img
        # self.resized_img = self.ResizeWithAspectRatio(Img.copy(), height=1280)
        utils.showTillkey("img", self.img)
        self.blured = cv.GaussianBlur(self.img.copy(),(5,5),3)#image, kernel_size and std_dev(σ)
        self.greyed = cv.cvtColor(self.blured.copy(), cv.COLOR_BGR2GRAY)
        self.edged = cv.Canny(self.greyed.copy(), 1, 180, apertureSize=3, L2gradient=False) #appearture is the sobel kernel and L2gradient give accurate edge to frame
        #                                         l_threshold, h_threshold
        # If the gradient magnitude of a pixel is below this threshold, the pixel is rejected as an edge. It’s useful for filtering out weak edges that may be caused by noise.
        # The upper threshold is used to detect strong edges. If the gradient magnitude of a pixel exceeds this value, it is considered an edge and will be included in the final edge map.

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8)) #structuring element for dialation close the larger gap between the edges.
        # dialete to close large edges. Dilation: Expands the white regions (foreground) in the image.
                                      # Erosion: Shrinks the white regions (foreground) in the image.
        self.edgeFilled = cv.dilate(self.edged, kernel, iterations=2)
          # Apply threshold to get binary image
        # _, binary = cv.threshold(self.edgeFilled, 128, 255, cv.THRESH_BINARY)
        # Invert the binary image
        # utils.showTillkey("edgeFilled", self.edgeFilled)

        # self.binary = cv.bitwise_not(self.edgeFilled)
        # Apply opening operation to remove small dots
        # opening = cv.morphologyEx(self.edgeFilled, cv.MORPH_OPEN, kernel, iterations=2)

        # utils.showTillkey("removed dots", opening)

        self.edgeFilled_coloured = cv.cvtColor(self.edgeFilled.copy(), cv.COLOR_GRAY2BGR) #contour_img
        # utils.showTillkey(self.edgeFilled_coloured)

        self.contours, self.heirarchy = cv.findContours(self.edgeFilled.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        self.applyContourOnEdgeFilled = cv.drawContours(self.edgeFilled_coloured.copy(), self.contours, -1, (0,0,255),3)
        print(1)
        utils.showTillkey("applyContourOnEdgeFilled", self.applyContourOnEdgeFilled)

        self.largestContour = max(self.contours, key=cv.contourArea)
        # Smooth the contour
        # epsilon = 0.002 * cv.arcLength(self.largestContour, True)
        # self.approx = cv.approxPolyDP(self.largestContour, epsilon, True)
        # Create a mask to store the filtered contours
        self.mask = np.zeros(self.edgeFilled.shape[:2], dtype=np.uint8)
        # Loop through each contour and draw it on the mask if it meets the area threshold
        for cnt in self.contours:
            if cv.contourArea(cnt) > 10000:
                cv.drawContours(self.mask, [cnt], -1, 255, -1)

        print(cv.contourArea(self.largestContour), "lar")
        self.maskForgetInsidePapper = self.mask #cv.drawContours(self.mask, dtype=np.uint8), [self.largestContour], -1, 255, -1)
        utils.showTillkey("maskForgetInsidePapper", self.maskForgetInsidePapper)
        # Define a kernel for morphological operations
        # kernel = np.ones((5, 5), np.uint8)

        # Erode to remove small noise
        # self.eroded_mask = cv.dilate(self.maskForgetInsidePapper, kernel, iterations=1)
        # -1: This parameter specifies which contours to draw. -1 means draw all contours in the list provided.
        # 255: This is the color to use for drawing the contours. In a grayscale image, 255 corresponds to white.
        # -1: This parameter specifies the thickness of the contour lines. -1 means fill the contour, rather than just drawing the outline.
        self.refinedEdge=cv.bitwise_and(self.edgeFilled, self.maskForgetInsidePapper)
        # The refinedEdge is retained only in the areas where self.maskForgetInsidePapper has non-zero values. 
        # it filters or refines self.edgeFilled based on the mask. 
        # This can be useful for excluding or including specific areas of interest in the image.
        utils.showTillkey("refinedEdge", self.refinedEdge)
        
        #-------bbox calc
        self.edgeCoordinatesForRefined = cv.findNonZero(self.refinedEdge) # find the coordinates of all non-zero pixels in a binary image. 
        # The minAreaRect function computes the minimum area rectangle that can enclose all the given points.
        # The result is a rotated rectangle, which is described by a tuple (center, (width, height), angle)
        self.CoordsAroundpapper = cv.minAreaRect(self.edgeCoordinatesForRefined)
        self.BoundboxPapperCoords=cv.boxPoints(self.CoordsAroundpapper)#obtaining 4 cornerpoints of rotated rectangle as [p1,p2,p3,p4]
        self.Bbox = np.int0(self.BoundboxPapperCoords) # np.int0() is similar to using np.astype(np.int32) but is a more concise way to perform the conversion.
        self.x, self.y = self.Bbox[0] # top left point of rectangle
        papper_width = self.Bbox[2][0]-self.x
        papper_height = self.Bbox[2][1]-self.y
        self.DebugPapperbox = cv.rectangle(self.img.copy(), (self.x, self.y), self.Bbox[2], (255,0,0), 3)
        print((self.x, self.y), self.Bbox[2])
        print(papper_width, papper_height)
        utils.showTillkey("DebugPapperbox", self.DebugPapperbox)

        # self.isolatedObject = 
        #--------------------------
        # self.smallObjContour = None
        # self.paper_axes = None
        # self.smallObjwthBbox = None
        self.imgWthPaperremoved = None
        self.remove_papper()
        self.edgeCoordinatesForRemovedPapper = cv.findNonZero(self.imgWthPaperremoved)
        self.CoordsAroundObject = cv.minAreaRect(self.edgeCoordinatesForRemovedPapper)
        self.BoundboxObjectCoords=cv.boxPoints(self.CoordsAroundObject)#obtaining 4 cornerpoints of rotated rectangle as [p1,p2,p3,p4]
        self.BboxObj = np.int0(self.BoundboxObjectCoords) # np.int0() is similar to using np.astype(np.int32) but is a more concise way to perform the conversion.
        self.xo, self.yo = self.BboxObj[0] # top left point of rectangle
        object_width = self.BboxObj[2][0]-self.xo
        object_height = self.BboxObj[2][1]-self.yo
        self.DebugObjectbox = cv.rectangle(self.img.copy(), (self.xo, self.yo), self.BboxObj[2], (255,0,0), 3)
        print((self.xo, self.yo), self.BboxObj[2])
        print(object_width, object_height)
        utils.showTillkey("DebugObjectbox", self.DebugObjectbox)

        self.width_cm = None
        self.height_cm = None
    #-- Calculations
        self.dimension_calc(papper_height, papper_width, object_height, object_width)
        print(f"width: {self.width_cm} , height: {self.height_cm}")
        self.display_dim(self.BboxObj,self.width_cm, self.height_cm, self.img.copy())
   

    def remove_papper(self):
        # convert2cv_u=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # contours, _= cv.findContours(self.edgeCoordinatesForRefined.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        max_contour = None
        for cnt in self.contours:
            area = cv.contourArea(cnt)
            if area>max_area:
                max_area=area
                max_contour= cnt
        for_mask = np.zeros_like(self.refinedEdge.copy())
        # for_mask = np.ones_like(self.refinedEdge.copy()) * 255  # Start with a white mask
        utils.showTillkey("for_mask", for_mask)

        mask = cv.drawContours(for_mask, [max_contour], -1, 255, thickness=cv.FILLED)
        utils.showTillkey("mask", mask)
        #Erossion
        kernel_size = 1000
        kernel  = np.ones((kernel_size, kernel_size), np.uint8)
        # Erode the mask to shrink the contour
        eroded_mask = cv.erode(mask, kernel, iterations=1)

        # Invert the eroded mask to prepare for removing the contour from the image
        inverted_mask = cv.bitwise_not(eroded_mask)
        utils.showTillkey("inverted_mask", inverted_mask)

        self.imgWthPaperremoved = cv.bitwise_and(self.refinedEdge, self.refinedEdge, mask=cv.bitwise_not(inverted_mask))
        utils.showTillkey("imgWthPaperremoved", cv.convertScaleAbs(self.imgWthPaperremoved))
 
    def dimension_calc(self, papper_height, papper_width, object_height, object_width):
       
        print("Object pixel width is" , object_width)
        print("Object pixel height is" , object_height)

        a_cm_pixel_width=papper_width/21
        print("a_cm_pixel_width", a_cm_pixel_width)
        a_cm_pixel_height=papper_height/29.7
        print("a_cm_pixel_height", a_cm_pixel_height)

        object_width_cm=object_width/a_cm_pixel_width
        object_height_cm=object_height/a_cm_pixel_height

        if object_height_cm<0:
            object_height_cm=object_height_cm*-1
        elif object_width_cm<0:
            object_width_cm=object_width_cm*-1

        self.width_cm = object_width_cm
        self.height_cm = object_height_cm
    
    def display_dim(self, Objbbox, width, height, frame1):
        x,y=Objbbox[0]
        x1,y1=Objbbox[2]
        width_text=f"width: {str(width)}"
        w_xpos=x+int(width*5.5714)//2
        w_ypos=y-10                                                             # fnt scale, color, thickness
        cv.putText(frame1, width_text, (w_xpos, w_ypos), cv.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 15)
    
        height_txt=f"height : {str(height)}"
        h_xpos=x1+100
        h_ypos=y1-int(height*4.8822)//2                                         # fnt scale, color, thickness
        cv.putText(frame1, height_txt, (h_xpos, h_ypos), cv.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 15)
        cv.rectangle(frame1, Objbbox[0], Objbbox[2], (255,0,0), 2)
        utils.showTillkey("Dimensions", frame1)
