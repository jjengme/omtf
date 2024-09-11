import numpy as np
from basicfunctions import cine2nparrall, cine2nparr, avg_cine, save_movie_nparr
from pyphantom import Phantom, utils, cine
import matplotlib.pyplot as plt
import cv2 as cv


ph = Phantom() # cine 파일을 불러올 때 항상해줘야됨

# # fname = 'Z:/03 exp/220126 ilasskorea/2bar 2x/2bar 106.cine'
# fname = '//203.253.185.128/kk/03 exp/240905/wi2.cine'
# frame = 0

# test = cine2nparr(fname, frame)

# # plt.figure
# # plt.imshow(test)
# # plt.show()

# cv.imshow('test',test)
# cv.waitKey()
# cv.destroyAllWindows()



# file_path = '//203.253.185.128/kk/03 exp/240905/wi2.cine'
# # cn = 'wi2.cine'

# # def avg_cine(cn): #fn의 모든 이미지를 평균함, 
# #     c = cine.Cine.from_filepath(cn) 
# #     nx, ny = c.resolution # 이미지의 가로 세로 픽셀수 얻기
# #     nz = c.range.last_image-c.range.first_image # frame 수
# #     ia = np.zeros((nx, ny), float) # 메모리 선배정(memory pre-allocation)

# #     for i in range(0, nz):
# #         ran = utils.FrameRange(c.range.first_image+i, c.range.first_image+i)    
# #         img = c.get_images(ran)[0,:,:] # 1) raw image
# #         ia = ia + img
# #     ia = ia / nz
# #     wi_a = ia/np.max(ia)
# #     return wi_a
    

# average_image =  avg_cine(file_path)
# wi_a = average_image/np.max(average_image)


# fname = '//203.253.185.128/kk/03 exp/240905/wi2.cine'
# frame = 0
# raw_i = cine2nparr(fname, frame)

# # nys = 
# # nye = 
# # nxs = 0
# # nxe = 
# # average_image = average_image[nys:nye, nxs:nxe]


# cv.imshow('raw_i', raw_i)
# cv.imshow('average_image', average_image)
# cv.waitKey()
# cv.destroyAllWindows()



fname = '//203.253.185.128/kk/03 exp/240905/case5.cine'
raw = cine2nparrall(fname)

save_movie_nparr(raw, 'case5.avi', 10) 