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

import time
start = int(round(time.time() * 1000))

# Open compress data file for reading
fd = open(infile, 'rb')

# Import module 'struct' to process bytes
import struct

# Read width and height
width, height = struct.unpack('@2i', fd.read(8))
out_size = width * height * 3

# Calculate block
w_new = (width + (8 - width % 8) % 8)
h_new = (height + (8 - height % 8) % 8)

w_block = w_new // 8
h_block = h_new // 8

# Checking data valid
expect_byte = w_block * h_block * 48 + 8
import io
fd.seek(0, io.SEEK_END)
in_size = fd.tell()
if expect_byte != in_size:
    print('File invalid, size not true!!!')
    sys.exit()
else:
    fd.seek(8)

import numpy as np
img_padded = np.zeros((h_new, w_new, 3), dtype=np.uint8)

import decompress as dcm

d_block = 0
x_pos = 0
y_pos = 0

count = 0

while x_pos != w_new and y_pos != h_new:
    pixel_cmp = np.frombuffer(fd.read(48), dtype=np.float128)
    d_block = dcm.decompress(pixel_cmp)
    img_padded[y_pos:y_pos+8,x_pos:x_pos+8,:] = d_block
    x_pos = x_pos + 8
    if x_pos == w_new:
        x_pos = 0
        y_pos = y_pos + 8

# Close file
fd.close()

# Clear padding component
img = img_padded[:height, :width, :]

end = int(round(time.time() * 1000))

print('Input size: {} bytes.'.format(in_size))
print('Output size: {} bytes.'.format(out_size))
print('Decompress ratio: {}%.'.format(np.round(out_size * 100 / (in_size - 8), 2)))
print('Decompress time: {} msec.'.format(end - start))

# Show image
import matplotlib.pyplot as plt
plt.imshow(img_padded)
plt.show()