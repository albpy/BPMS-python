import  cv2 as cv

class utils:
    def show(name, img):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(2000)
        cv.destroyAllWindows()

    def showTillkey(name, img):
        while True:
            cv.namedWindow(name, cv.WINDOW_NORMAL)
            cv.imshow(name, img)
            if (cv.waitKey(25)&0xFF) ==ord('q'):
                cv.destroyAllWindows()
                break
            