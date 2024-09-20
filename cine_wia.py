import numpy as np
from basicfunctions import cine2nparrall, cine2nparr, avg_cine, save_movie_nparr
from pyphantom import Phantom, utils, cine
import matplotlib.pyplot as plt
import cv2 as cv


ph = Phantom() # cine 파일을 불러올 때 항상해줘야됨

file_path = 'Z:/03 exp/240905/wi2.cine'

cn = 'wi2.cine'   #cn : cine 파일 이름

def avg_cine(cn): #fn의 모든 이미지를 평균함, 
    c = cine.Cine.from_filepath(cn) 
    nx, ny = c.resolution # 이미지의 가로 세로 픽셀수 얻기
    nz = c.range.last_image-c.range.first_image # frame 수
    ia = np.zeros((nx, ny), float) # 메모리 선배정(memory pre-allocation)

    for i in range(0, nz):
        ran = utils.FrameRange(c.range.first_image+i, c.range.first_image+i)    
        img = c.get_images(ran)[0,:,:] # 1) raw image
        ia = ia + img
    ia = ia / nz
    wi_a = ia/np.max(ia)
    return wi_a
    

average_image =  avg_cine(file_path)
wi_a = average_image/np.max(average_image)


# 원본 이미지와 white average 이미지 비교
# fname = 'Z:/03 exp/240905/wi2.cine'
# frame = 0
# raw = cine2nparr(fname, frame)

# cv.imshow('raw',raw)
# cv.imshow('wi_a',wi_a)
# cv.waitKey()
# cv.destroyAllWindows() 
