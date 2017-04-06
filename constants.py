import numpy as np

SRC = np.float32([[580, 460],[700, 460],[1040, 680],[260, 680]])
DST = np.float32([[260, 0],[1040, 0],[1040, 720],[260, 720]])
    

RELEVANT_HIST = 5

# meters per pixel in y dimension
YM_PP = 30. / 720
# meters per pixel in x dimension
XM_PP = 3.7 / 700

