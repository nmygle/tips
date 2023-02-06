from math import ceil
from PIL import Image


def resize(img, max_size=768):
    w, h = img.size
    if w > h:
        h_r = ceil(max_size / w * h)
        img = img.resize([max_size, h_r])
    else:
        w_r = ceil(max_size / h * w)
        img = img.resize([w_r, max_size])
    return img


def padding(img, low_size=512, max_size=768):
    w, h = img.size
    background_color = (0, 0, 0)
    if w > h:
        if h < low_size:
            new_height = low_size
        else:
            new_height = max_size
        img_new = Image.new(img.mode, (w, new_height), background_color)
        diff = new_height - h
        img_new.paste(img, (0, diff // 2))
    else:
        if w < low_size:
            new_width = low_size
        else:
            new_width = max_size
        img_new = Image.new(img.mode, (new_width, h), background_color)
        diff = new_width - w
        img_new.paste(img, (diff // 2, 0))
    return img_new
