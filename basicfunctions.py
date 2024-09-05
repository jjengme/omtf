import sys, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import fft2, ifft2
from pyphantom import Phantom, utils, cine
from skimage import data # m-Otsu를 위해 필요
from skimage.filters import threshold_multiotsu # m-Otsu를 위해 필요

# rut 폴더에 있는 n+1번째 jpg 이미지를 행렬로 불러옴
def import_raw(rut, n):
    file_list = glob.glob(rut + '*.jpg')
    img = cv2.imread(file_list[n], cv2.IMREAD_GRAYSCALE) 
    return img

# rut 폴더에 있는 n+1번째 tif 이미지를 행렬로 불러옴
def import_raw(rut, n):
    file_list = glob.glob(rut + '*.tif')
    img = cv2.imread(file_list[n], cv2.IMREAD_GRAYSCALE) 
    return img


def avg_img(path): #path의 모든 이미지를 평균함
    file_list = glob.glob(path + '*.jpg') # white image 위치 path.txt파일의 제일 첫줄에 적혀있음
    nx, ny, _ = cv2.imread(file_list[0]).shape # 이미지의 가로 세로 픽셀수 얻기
    ia = np.zeros((nx, ny), float) # 메모리 선배정(memory pre-allocation)

    for i in file_list:
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        ia = ia + img
    ia = ia / len(file_list)
    return ia

def rel_std(cine, ref): # cine(파일명+경로)이미지 - ref(2d np array)
    dum = ref[:,:,np.newaxis] - cine2nparrall(cine)#
    rstd = np.std(dum, 2)/np.mean(dum, 2)
    return rstd

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def pixel_binning(image, binning_factor):
    h, w = image.shape[:2]
    
    # Calculate new dimensions after binning
    new_h = h // binning_factor
    new_w = w // binning_factor
    
    # Reshape the image to facilitate binning
    reshaped_image = image[:new_h * binning_factor, :new_w * binning_factor].reshape(new_h, binning_factor, new_w, binning_factor)
    
    # Take the average within each bin
    binned_image = reshaped_image.mean(axis=(1, 3)).astype(np.uint8)
    
    return binned_image

# white image correction 수행
# img: raw img, rutA: white image가 있는 폴더
# nys: 크롭 시작행, nye: 크롭 종료행, nxs: 크롭 시작열, nxe: 크롭 종료열
def wic(img, rutA, nys, nye, nxs, nxe):
    wi = avg_img(rutA)
    wi = wi/np.max(wi) 
    wi = wi[nys:nye, nxs:nxe] # image crop
    c = np.array(np.round(img/wi), dtype=np.uint8) # white image correction
    c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX) # stretching
    return c

def lap_spa(img):
    lap = cv2.Laplacian(img, cv2.CV_32F, 3) # 1) laplacian
    lap = cv2.normalize(lap, None, -1, 1, cv2.NORM_MINMAX) # 7) normalization
    sha = img - 255 * lap # 2) sharpening
    return lap, sha

def lap_fre(img):
    F = np.fft.fftshift(np.fft.fft2(img/255))# freq laplacian filter
    P,Q = F.shape
    H = np.zeros((P,Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            H[u,v] = -4*np.pi*np.pi*((u-P/2)**2 + (v-Q/2)**2)

    # Laplacian image
    Lap = np.fft.ifftshift(H * F)
    Lap = np.real(np.fft.ifft2(Lap))
    sha = img/255 - cv2.normalize(Lap, None, -1, 1, cv2.NORM_MINMAX) # 7) normalization
    sha = np.clip(sha, 0, 1)

    return sha

# 2) Laplacian filter in spatial domain using smoothed sobel edge, Gonzales book ch 4
def laplacian(img): 
    Lap, Sha = lap_spa(img) #Lap: Laplacian, Sha: sharpened in spatial domain
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0) # float 형태의 미분값을 저장
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    Sob = cv2.magnitude(dx, dy) # 3) Edge, sobel gradient
    del dx, dy
    Sobs = cv2.filter2D(Sob, -1, np.ones((5,5),np.float32)/25) # 4) Edge smoothed, sobel gradient smoothed 5 by 5 box filter 
    del Sob
    mask = cv2.multiply(Lap, Sobs, dtype = cv2.CV_32F)  # 5) mask: laplacian * smoothed sobel
    img_sha = np.clip(img - mask, 0, 255).astype(np.float32)
    # img_sha = np.clip(img - mask, 0, 255).astype(np.uint8)
    return img_sha

##################################################################
# c    i    n    e
##################################################################
# cn: 경로를 포함한 cine 파일 이름

# cine 파일의 해상도 구하기
def cine_resolution(cn): # cn: cine 파일 이름
    c = cine.Cine.from_filepath(cn) 
    nx, ny = c.resolution # 이미지의 가로 세로 픽셀수 얻기
    return nx, ny

# cine 파일의 n+1번째 프레임을 불러들여 2차원 넘파이 배열(np2arr)로 반환함
def cine2nparr(cn, n): # cn: cine 파일 이름(경로포함), n: 프레임 넘버 (0이 첫번째 프레임)
    c = cine.Cine.from_filepath(cn) 
    ran = utils.FrameRange(c.range.first_image+n, c.range.first_image+n)    
    np2arr = c.get_images(ran)[0,:,:]
    return np2arr

# cine 파일을 불러들여 전체를 평균함
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
    return ia

# cine 파일을 불러들여 전체를 3차원 넘파이 배열(np3arr)로 반환함
def cine2nparrall(cn): # cn: cine 파일 이름
    c = cine.Cine.from_filepath(cn) 
    nx, ny = c.resolution # 이미지의 가로 세로 픽셀수 얻기
    nz = c.range.last_image-c.range.first_image # frame 수
    np3arr = np.zeros((nx, ny, nz), dtype = np.uint8) # 반환할 3차원 넘파이 배열
    for i in range(0, nz):
        ran = utils.FrameRange(c.range.first_image+i, c.range.first_image+i)    
        np3arr[:,:,i] = c.get_images(ran)[0,:,:] # 1) raw image
    return np3arr

##################################################################################
#          이미지 후처리
##################################################################################

# 넘파이 어레이를 image processing 수행
def nparr_imgproc(np2arr): #  2차원 array
    sha = laplacian(np2arr)
    # 3) Binarization
    # m-otsu
    thresholds = threshold_multiotsu(sha, 3) # 영역을 세군데로 나눔.
    regions = np.digitize(sha, bins=thresholds)
    bi = np.where(sha < thresholds[-1], sha, 255) # 이진화 
    # bi = np.where(sha < thresholds[-1], sha, thresholds[-1]) # 이진화 
    return sha

# .cine 파일의 n번째 프레임을 넘파이 어레이로 불러와서 image processing 수행
def import_cine(fn, n): # fn: cine file의 이름, n: 프레임 수
    c = cine.Cine.from_filepath(fn)  # cine 파일의 모든 정보가 파이썬으로 import
    test_range = utils.FrameRange(c.range.first_image+n, c.range.first_image+n) 
    raw = c.get_images(test_range)[0,:,:]  # raw
    # 2) Laplacian filter in spatial domain using smoothed sobel edge
    sha = laplacian(raw)
    # 3) Binarization
    # m-otsu
    thresholds = threshold_multiotsu(sha, 3) # 영역을 세군데로 나눔.
    regions = np.digitize(sha, bins=thresholds)
    bi = np.where(sha < thresholds[-1], sha, 255) # 이진화 
    # bi = np.where(sha < thresholds[-1], sha, thresholds[-1]) # 이진화 
    return sha


##################################################################################
#          동영상
##################################################################################

# 3차원 넘파이 배열을 cv2.imshow()를 이용해 동영상으로 화면에 재생
def play_nparr(np3arr, fps): #np3arr: 3차원 넘파이 배열, fps: 재생속도
    for k in range(np3arr.shape[2]):
        cv2.imshow('Wanna stop? ESC', np3arr[:,:,k])
        # cv2.resizeWindow(winname='Wanna stop? ESC', width=200, height=150)
        if cv2.waitKey(int(1000/fps)) == 27:
            break

# 3차원 넘파이 배열을 plt.imshow()를 이용해 동영상으로 화면에 재생
def play_nparr_plt(np3arr, fig, ax): #np3arr: 3차원 넘파이 배열

    ims = []

    for i in range(np3arr.shape[2]):
        im = ax.imshow(np3arr[:,:,i], animated=True, cmap = 'jet', vmin = 0, vmax = 100)
        ax.axis('off')
        # ax.set_title(str(i) + 'th frame')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False, repeat = False,
                                repeat_delay=1000)
    ani.save('movie', fps = 10)
    return ani


# # 3차원 넘파이 배열을 plt.imshow()를 이용해 동영상으로 화면에 재생
# def play_nparr_plt(np3arr): #np3arr: 3차원 넘파이 배열
#     plt.show()
#     for k in range(np3arr.shape[2]):
#         plt.imshow(np3arr[:,:,k], cmap = 'jet')
#         plt.title(str(k))
#         cv2.waitKey()

# 3차원 넘파이 배열을 opencv를 이용하여 동영상으로 저장
def save_movie_nparr(np3arr, filename, fps): # np3arr: 3차원 넘파이 배열, filename: 저장할 파일이름, fps: frame per sec
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    vny = np3arr.shape[1];vnx = np3arr.shape[0]
    out = cv2.VideoWriter(filename, fourcc, fps, (vny,vnx)) # save movie
    # Loop for each frame
    for i in range(np3arr.shape[2]):
        frame = cv2.cvtColor(np3arr[:,:,i], cv2.COLOR_GRAY2BGR)
        out.write(frame)


 