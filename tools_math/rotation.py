import math
import numpy as np

def rotate2d(input_, angle, axis):
    """
    input
        input_: 入力ベクトル（2次元）
        angle: 回転角（degree）
        axis: 回転軸
    """
    input_ = np.array(input_)
    axis = np.array(axis)
    rad = math.radians(angle)
    x = input_ - axis
    xt0 = np.cos(rad) * x[0] - np.sin(rad) * x[1] + axis[0]
    xt1 = np.sin(rad) * x[0] + np.cos(rad) * x[1] + axis[1]
    return np.array([xt0, xt1])
  
