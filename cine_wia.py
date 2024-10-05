import numpy as np
from basicfunctions import cine2nparrall, cine2nparr, avg_cine, save_movie_nparr
from pyphantom import Phantom, utils, cine
import matplotlib.pyplot as plt
import cv2 as cv


ph = Phantom() # cine 파일을 불러올 때 항상해줘야됨

file_path = 'Z:/03 exp/240905/wi2.cine'

cn = 'wi2.cine'   #cn : cine 파일 이름

# Defenition of avg_cine(cn)
# def avg_cine(cn): #fn의 모든 이미지를 평균함, 
#     c = cine.Cine.from_filepath(cn) 
#     nx, ny = c.resolution # 이미지의 가로 세로 픽셀수 얻기
#     nz = c.range.last_image-c.range.first_image # frame 수
#     ia = np.zeros((nx, ny), float) # 메모리 선배정(memory pre-allocation)

#     for i in range(0, nz):
#         ran = utils.FrameRange(c.range.first_image+i, c.range.first_image+i)    
#         img = c.get_images(ran)[0,:,:] # 1) raw image
#         ia = ia + img
#     ia = ia / nz
#     wi_a = ia/np.max(ia)
#     return wi_a
    
average_image =  avg_cine(file_path)
whimg = average_image/np.max(average_image)

# 원본 이미지와 white average 이미지 비교
# fname = 'Z:/03 exp/240905/wi2.cine'
# frame = 0
# raw = cine2nparr(fname, frame)

# cv.imshow('raw',raw)
# cv.imshow('whimg',whimg)
# cv.waitKey()
# cv.destroyAllWindows() 

# 크롭하기 위한 좌표
nys = 80
nye = 1170
nxs = 0
nxe = 360

whimg = whimg[nys:nye, nxs:nxe]

path = 'Z:/03 exp/240905/case5.cine'
frame = 0

raw = cine2nparr(path, frame)
raw = raw[nys:nye, nxs:nxe]

#  white image correction
wic = np.array(np.round(raw/whimg), dtype=np.uint8) 
wic = cv.normalize(wic, None, 0, 255, cv.NORM_MINMAX) # stretching

# clahe 객체 정의
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# 2. 명암비 강화
he = cv.equalizeHist(wic) # 2-1 히스토그램 평활화
cl = clahe.apply(wic) # 2-2 CLAHE
la = wic - 0.5 * cv.Laplacian(wic, cv.CV_16S, 3) # 2-3 laplacian sharpening
la = cv.normalize(la, None, 0, 255, cv.NORM_MINMAX)
