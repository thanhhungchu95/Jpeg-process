# Checking usage syntax
import sys

if len(sys.argv) != 2:
    print("Argument invalid!")
    print("Usage: python3 jpeg-compressor.py [input]")
    sys.exit()

infile = sys.argv[1]

# Checking file is exist
import os.path

if not os.path.isfile(infile):
    print("Input file not exist!")
    sys.exit()

# Checking file type
import imghdr

if imghdr.what(infile) == None:
    print("Input file is not a supported image file!");
    sys.exit()

import time
start = int(round(time.time() * 1000))

# Read image data for compress
import scipy.ndimage as sn

img = sn.imread(infile, mode='RGB')

# Get image size
width = len(img[0])
height = img.size//width//3

in_size = 8 + width * height * 3

# Open file for writing
outfile = infile + '.jc'
fd = 0
if os.path.isfile(outfile):
    fd = open(outfile, 'wb')
else:
    fd = open(outfile, 'xb')

out_size = 0

# First, write original size
import struct
isize = struct.pack('@2i', width, height)
fd.write(isize)
out_size += 8

# Prepare for cutting image data to 8x8 block
# Other work like dct, quantization ... do after cutting block
# We are working with block only, after cutting it, we are use pre result for post work parameter

import numpy as np

# Calculate new size
# New size no need to write, because padding <= 8, we'll use func ceil() to round it
w_new = width + (8 - width % 8) % 8
h_new = height + (8 - height % 8) % 8

# Padding zero value to original image data to have 8t * 8k pixel with t = width/8 and k = height/8
img_padded = np.zeros((h_new, w_new, 3), dtype=img.dtype)
img_padded[:height, :width, :] = img

# Initialize x_pos and y_pos
x_pos = 0
y_pos = 0

# Initialize temp bytes (compress image data of 8x8 block)
b_cmp = 0

import compress as cm

while y_pos < h_new:
    while x_pos < w_new:
        c_block = img_padded[y_pos:y_pos + 8, x_pos:x_pos + 8]
        b_cmp = cm.compress(c_block)
        # Write pixel to file
        # b_cmp is a list with three compress component: Y, Cb, Cr
        fd.write(b_cmp[0].tobytes())
        fd.write(b_cmp[1].tobytes())
        fd.write(b_cmp[2].tobytes())
        out_size += 48
        x_pos = x_pos + 8
    x_pos = 0
    y_pos = y_pos + 8

fd.close()

end = int(round(time.time() * 1000))

print('Input file: {}, size = {} bytes'.format(infile, in_size))
print('Output file: {}, size = {} bytes.'.format(outfile, out_size))
print('Compress ratio: {}%.'.format(np.round(out_size * 100 / in_size, 2)))
print('Compress time: {} msec.'.format(end - start))