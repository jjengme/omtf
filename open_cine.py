import numpy as np
from basicfunctions import cine2nparrall, cine2nparr, avg_cine, save_movie_nparr
from pyphantom import Phantom, utils, cine
import matplotlib.pyplot as plt
import cv2 as cv


ph = Phantom() # cine 파일을 불러올 때 항상해줘야됨

ex)
fname = '//203.253.185.128/kk/03 exp/240905/wi2.cine'
frame = 0

test = cine2nparr(fname, frame)

cv.imshow('test',test)
cv.waitKey()
cv.destroyAllWindows()
