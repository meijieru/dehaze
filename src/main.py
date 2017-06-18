from __future__ import print_function

import argparse
import cv2
import os

import utils
import dehaze

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="index for single input image")
parser.add_argument("--t_min", type=float, default=0.25, help="minimum transmission rate")
parser.add_argument("--window", type=int, default=15, help="window size of dark channel")
parser.add_argument("--radius", type=int, default=80, help="radius of guided filter")
parser.add_argument("--omega", type=float, default=0.95, help="percantage of haze to be removed")
parser.add_argument("--refine", type=bool, default=True, help="whether to refine the transmission estimated")

opt = parser.parse_args()

# path cal
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_ROOT = os.path.dirname(FILE_DIR)
DST_ROOT = os.path.join(BASE_ROOT, 'res')
utils.assure_dir(DST_ROOT)

src_path = os.path.abspath(opt.input)
dst_name = utils.format(
    os.path.split(src_path)[1], opt.window, opt.radius, opt.omega, opt.t_min, opt.refine)
dst_path = os.path.join(DST_ROOT, dst_name)

dehazer = dehaze.DarkPriorChannelDehaze(
    wsize=opt.window, radius=opt.radius, omega=opt.omega,
    t_min=opt.t_min, refine=opt.refine)

src_img = cv2.imread(src_path, cv2.IMREAD_COLOR)
img_dehaze = dehazer(src_img)
cv2.imwrite(dst_path, img_dehaze)

print('Saved to: {}'.format(dst_path))
