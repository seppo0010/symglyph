from itertools import combinations
import string

import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont, ImageChops
from skimage.color import rgb2gray, rgba2rgb
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def draw_image(text, font):
    txt = Image.new("RGBA", (1024, 1024), (255,255,255,0))
    d = ImageDraw.Draw(txt)
    d.text((0,0), text, fill=(0,0,0), font=font)
    txt = trim(txt)
    return txt

def similitude(text1, text2, font=None):
    if font is None:
        font = ImageFont.truetype("Roboto/Roboto-Regular.ttf", 96)
    im1 = numpy.asarray(draw_image(text1, font=font))
    im2 = numpy.asarray(draw_image(text2, font=font))
    im2 = cv2.resize(im2, [im1.shape[1], im1.shape[0]])
    return ssim(rgb2gray(rgba2rgb(im1)), rgb2gray(rgba2rgb(im2)))

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

data = {}
for c1, c2 in tqdm(list(combinations(string.printable, 2))):
    s = similitude(c1, c2)
    data[(c1, c2)] = s
    data[(c2, c1)] = s

print(data)
