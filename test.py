# import numpy as np
# from basicfunctions import cine2nparrall, cine2nparr, avg_cine, save_movie_nparr
# from pyphantom import Phantom, utils, cine
# import matplotlib.pyplot as plt
import cv2 as cv


src = cv.imread('capture.png')
src = cv.resize(src,(0,0),fx= 0.6,fy=0.6)

cv.imshow('src',src)
cv.waitKey()
cv.destroyAllWindows()