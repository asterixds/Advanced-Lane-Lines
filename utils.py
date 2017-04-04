import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2, os, re


"""display the image"""
def showimage(img):
    plt.imshow(img)
    plt.show()

"""arrange images in a grid"""
def visualise_in_squaregrid(data, title="Image grid square", output_file=None):
    n = int(np.ceil(np.sqrt(len(data))))
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(n, n, wspace=0.0)
    ax = [plt.subplot(gs[i]) for i in range(n*n)]
    gs.update(hspace=0)
    for i,im in enumerate(data):
        ax[i].imshow(im)
        ax[i].axis('off')
    if output_file:
        fig.savefig(output_file)
    plt.show()


"""various image transforms to threshold the image and generate a binary image of candidate pixels for lane detection"""
# Gradient and Color thresholds
def scale(img, factor=255.0):
    scale_factor = np.max(img)/factor
    return (img/scale_factor).astype(np.uint8)

def threshold(img, thresh = (0,255)):
    binary_img = np.zeros_like(img)
    binary_img[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_img

def gradients_abs(gray,sobel_kernel=15 ):
    gx_abs = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    gy_abs = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    return gx_abs, gy_abs

def mag_thresh( gx_abs, gy_abs, thresh=(50,255)):
    gmag = np.sqrt(gx_abs**2 + gy_abs**2)
    gmag= scale(gmag)
    return threshold(gmag,thresh)

def dir_thresh( gx_abs, gy_abs, thresh=(0.7,1.2)):
    gdir = np.arctan2(gy_abs, gx_abs)
    return threshold(gdir, thresh)

def rgb_decompose(img):
    rgb = img
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    return r,g,b

def hls_decompose(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h = hsv[:,:,0]
    l = hsv[:,:,1]
    s = hsv[:,:,2]
    return h,l,s
 
def filter_colours( img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    w_min = np.array([0, 0, 200], np.uint8)
    w_max = np.array([255, 30, 255], np.uint8)
    w_mask = cv2.inRange(img, w_min, w_max)
    
    y_min = np.array([15, 100, 120], np.uint8)
    y_max = np.array([80, 255, 255], np.uint8)
    y_mask = cv2.inRange(img, y_min, y_max)

    binary_img = np.zeros_like(img[:, :, 0])
    binary_img[((y_mask != 0) | (w_mask != 0))] = 1
    return binary_img
    
def filter_colours2(img):
    # Filter white and yellow colours
    r,g,b = rgb_decompose(img)
    h,l,s = hls_decompose(img)
    w = threshold(s,(200,255))
    y = np.logical_and(threshold(r,(200,255)), threshold(g,(200,255)))
    mask = np.logical_or(w,y)
    binary_img = np.zeros_like(img[:, :, 0])
    binary_img[mask] =1
    return binary_img


"""lane detection utilities"""
def draw_boundaries(img, p1, p2, color=(255,255,0), thickness = 5):
    cv2.rectangle(img, p1, p2, color, thickness)


def indices_within_win_limits(nonzero_x, nonzero_y, x_lo, y_lo, x_hi, y_hi):
    cond1 = (nonzero_x > x_lo)
    cond2 = (nonzero_x < x_hi)
    cond3 = (nonzero_y > y_lo)
    cond4 = (nonzero_y < y_hi)
    return (cond1 & cond2 & cond3 & cond4 ).nonzero()[0]     

def nonzero(wb):
    nonzero = np.where(wb != 0)
    return np.array(nonzero[0]), np.array(nonzero[1])

"""Sliding window class to declutter the code to progress the sliding window"""
class SlidingWindow:
    def __init__(self, margin,y_hi,cx,h):
        self.margin = margin
        self.h = h
        self.cx = cx
        self.y_hi = y_hi
        self.y_lo = y_hi-h
        self.x_lo = cx - margin
        self.x_hi = cx + margin
    
    def coordinates(self ):
        return self.x_lo, self.y_lo, self.x_hi, self.y_hi, self.cx
    
    def move(self, cx ):
        self.cx = cx
        self.x_lo = cx - self.margin
        self.x_hi = cx + self.margin
        self.y_hi -= self.h
        self.y_lo -= self.h
        return self.coordinates()
    