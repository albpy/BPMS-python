import cv2 as cv
from utils import *
from pxl_cntrl import *

path = "frames/redmi_7A.jpg"
raw_img =  cv.imread(path)

if __name__ == "__main__":
    # utils.show(raw_img)
    image = Image(raw_img)

    # image.detectEdgeAndContour()
    # utils.show(contour_img)
    # image.eradicate_large_contour()

    # image.min_arearect()

    # image.drawrect()

    # image.remove_papper()

    # image.detect()

    # image.dimension_calc(image.paper_axes, image.obj_cords)

    # image.display_dim(image.obj_cords, image.width, image.height, image.img.copy())