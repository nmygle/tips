from PIL import Image
from math import floor

def get_keep_aspect_size(w0, h0, new_size):
    if w0 > h0:
        w1 = new_size
        h1 = floor(h0 * (new_size / w0))
    else:
        h1 = new_size
        w1 = floor(w0 * (new_size / h0))
    return w1, h1


if __name__ == "__main__":
    image = Image.open("test.png")
    w0, h0 = image.size
    new_size = 512
    w1, h1 = get_keep_aspect_size(w0, h0, new_size)
    print(w0, h0)
    print(w1, h1)
    image = image.resize((w1, h1))
