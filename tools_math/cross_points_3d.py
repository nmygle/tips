import numpy as np

def get_cross_point(p0, p1, p2, x0, k_vec):
    """
    Args:
        p0, p1, p2 (np.ndarray) : 平面を構成する３点
        x0, k_vec (np.ndarray)　: 直線の視点とベクトル
    Returns:
        cross_point (np.ndarray) : 平面と直線の交点
    """
    # 平面の法線ベクトル
    cross = np.cross(p1 - p0, p2 - p0)
    # 平面の方程式の右辺(ax+by+cz=h)
    h = np.inner(cross, p0)
    # 直線(x0_vec+t*k_vecと平面の交点（ax+by+cz=h）の交点となるt
    t = (h - np.inner(x0, cross)) / np.inner(k_vec, cross)
    # 交点の座標
    cross_point = x0 + t * k_vec
    return cross_point
