"""
Decompress utilities file
"""
import numpy as np				# Import to process numpy array
import scipy.fftpack as sf		# Import to process 2-dimension idct

# Default transform matrix from YCbCr to RGB
rgb_mat = np.asmatrix([[1.0, 1.0, 1.0], [0.0, -0.3441, 1.772], [1.402, -0.7141, 0.0]])
rgb_pad = np.asmatrix([0, -128, -128])

# Default Quantization table
quant_tbl = np.array([[16, 11, 10, 16, 24, 40, 51, 61], \
                      [12, 12, 14, 19, 26, 58, 60, 55], \
                      [14, 13, 16, 24, 40, 57, 69, 56], \
                      [14, 17, 22, 29, 51, 87, 80, 62], \
                      [18, 22, 37, 56, 68, 109, 103, 77], \
                      [24, 35, 55, 64, 81, 104, 113, 92], \
                      [49, 64, 78, 87, 103, 121, 120, 101], \
                      [72, 92, 95, 98, 112, 100, 103, 99]])

# Probability interval of arithmetic coding interval
interval={-32:[0.0,0.001],-31:[0.001,0.002],-30:[0.002,0.003],-29:[0.003,0.004],
          -28:[0.004,0.005],-27:[0.005,0.006],-26:[0.006,0.007],-25:[0.007,0.008],
          -24:[0.008,0.009],-23:[0.009,0.01],-22:[0.01,0.011],-21:[0.011,0.012],
          -20:[0.012,0.013],-19:[0.013,0.014],-18:[0.014,0.015],-17:[0.015,0.016],
          -16:[0.016,0.017],-15:[0.017,0.022],-14:[0.022,0.027],-13:[0.027,0.032],
          -12:[0.032,0.037],-11:[0.037,0.042],-10:[0.042,0.047],-9:[0.047,0.052],
          -8:[0.052,0.057],-7:[0.057,0.069],-6:[0.069,0.081],-5:[0.081,0.093],
          -4:[0.093,0.105],-3:[0.105,0.155],-2:[0.155,0.205],-1:[0.205,0.325],
          0:[0.325,0.676],1:[0.676,0.796],2:[0.796,0.846],3:[0.846,0.896],
          4:[0.896,0.908],5:[0.908,0.92],6:[0.92,0.932],7:[0.932,0.944],
          8:[0.944,0.949],9:[0.949,0.954],10:[0.954,0.959],11:[0.959,0.964],
          12:[0.964,0.969],13:[0.969,0.974],14:[0.974,0.979],15:[0.979,0.984],
          16:[0.984,0.985],17:[0.985,0.986],18:[0.986,0.987],19:[0.987,0.988],
          20:[0.988,0.989],21:[0.989,0.99],22:[0.99,0.991],23:[0.991,0.992],
          24:[0.992,0.993],25:[0.993,0.994],26:[0.994,0.995],27:[0.995,0.996],
          28:[0.996,0.997],29:[0.997,0.998],30:[0.998,0.999],31:[0.999,1.0]}

# Global variable, store previous value of dc component for each Y, Cb, Cr
y_pre_dc = 0
cb_pre_dc = 0
cr_pre_dc = 0


def decompress(pixel_cmp):
    """ Decompress image compress data function
    :param
        pixel_cmp: Include three float value which is code for Y, Cb, Cr
    : return 8x8 block of image data
    """

    # Get Y, Cb, Cr code-word
    y_code, cb_code, cr_code = pixel_cmp
    
    # Decode each component with arithmetic coding algorithm, output is array of dc and ac
    y_decode, cb_decode, cr_decode = decode(y_code), decode(cb_code), decode(cr_code)
    
    # Convert dc and ac to zig-zag array
    y_zz, cb_zz, cr_zz = get_zigzag(y_decode, cb_decode, cr_decode)
    
    # Convert zig-zag array to pre quantization value
    y_quant = get_quant(y_zz)
    cb_quant = get_quant(cb_zz)
    cr_quant = get_quant(cr_zz)
    
    # Convert pre quantization value to pre dct value (using idct method)
    y_bl = np.round(sf.idct(y_quant, norm='ortho'))
    cb_bl = np.round(sf.idct(cb_quant, norm='ortho'))
    cr_bl = np.round(sf.idct(cr_quant, norm='ortho'))
    
    # Return 8x8 block image
    return yc2rgb(y_bl, cb_bl, cr_bl)


def findIndex(code):
    """ Find range for an symbol of list dc and ac component 
        which is output of Arithmetic Coding encode algorithm
    :param
        code: A float value
    :return range of symbol
    """
    for i in interval:
        if code >= interval[i][0] and code < interval[i][1]:
            return i
    return 0


def yc2rgb(y_bl, cb_bl, cr_bl):
    """ YCbCr2RGB function
    :param
        y_bl: 8x8 block of Y value
        cb_bl: 8x8 block of Cb value
        cr_bl: 8x8 block of Cr value
    :return 8x8 block of [R, G, B]
    """
    rgb_bl = np.zeros((8, 8, 3), dtype=np.uint8)
    
    for i in range(8):
        for j in range(8):
            yc_cmp = np.asmatrix([y_bl[i][j], cb_bl[i][j], cr_bl[i][j]]) + 128
            rgb_cmp = (yc_cmp-rgb_pad)*rgb_mat
            rgb_bl[i][j] = np.array(rgb_cmp, dtype=np.uint8)
    return rgb_bl


def get_dc_and_ac(dcm_cmp):
    """ Get DC and AC value
    :param
        dcm_cmp: array of dc and ac component
    :return DC(a value) and AC(a list) value
    """
    dc = dcm_cmp[0]
    ac = dcm_cmp[1:]
    return dc, ac


def get_zigzag(y_cmp, cb_cmp, cr_cmp):
    """ Get 1x64 zig-zag block
    :param
        y_cmp: Item of Y
        cb_cmp: Item of Cb
        cr_cmp: Item of Cr
    :return
        Three 1x64 block of each component Y, Cb, Cr
    """
    y_dc, y_ac = get_dc_and_ac(y_cmp)
    cb_dc, cb_ac = get_dc_and_ac(cb_cmp)
    cr_dc, cr_ac = get_dc_and_ac(cr_cmp)
    global y_pre_dc, cb_pre_dc, cr_pre_dc
    y_dc, cb_dc, cr_dc = y_dc + y_pre_dc, cb_dc + cb_pre_dc, cr_dc + cr_pre_dc
    y_pre_dc, cb_pre_dc, cr_pre_dc = y_dc, cb_dc, cr_dc
    return process_zigzag(y_dc, y_ac), process_zigzag(cb_dc, cb_ac), process_zigzag(cr_dc, cr_ac)


def process_zigzag(dc, ac):
    """ Calculate zig-zag block from dc and ac component
    :param
        dc: DC component
        ac: Ac component (list)
    :return 1x64 block
    """
    lst = [dc]
    for i in range(len(ac)):
        if i % 2 == 0:
            for j in range(i):
                lst.append(0)
        else:
            lst.append(ac[i])
    for i in range(64 - len(lst)):
        lst.append(0)
    return np.array(lst, dtype=np.int8)


def get_quant(zz):
    """ Get post quantization block
    :param 
        zz: 1x64 zig-zag block
    :return An 8x8 block post-quantization
    """
    return np.array([[zz[0], zz[1], zz[5], zz[6], zz[14], zz[15], zz[27], zz[28]], 
                     [zz[2], zz[4], zz[7], zz[13], zz[16], zz[26], zz[29], zz[42]], 
                     [zz[3], zz[8], zz[12], zz[17], zz[25], zz[30], zz[41], zz[43]], 
                     [zz[9], zz[11], zz[18], zz[24], zz[31], zz[40], zz[44], zz[53]], 
                     [zz[10], zz[19], zz[23], zz[32], zz[39], zz[45], zz[52], zz[54]], 
                     [zz[20], zz[22], zz[33], zz[38], zz[46], zz[51], zz[55], zz[60]], 
                     [zz[21], zz[34], zz[37], zz[47], zz[50], zz[56], zz[59], zz[61]], 
                     [zz[35], zz[36], zz[48], zz[49], zz[57], zz[58], zz[62], zz[63]]], 
                     dtype=np.int8) * quant_tbl


def decode(c_value):
    """ Decode function
    :param
        c_value: code of list dc and ac (a float value)
    :return List of DC and AC component
    """
    lst = []
    while True:
        index = findIndex(c_value)
        if index < 0 and len(lst) % 2 != 0:
            lst.append(0)
            lst.append(0)
        else:
            lst.append(index)
            c_value = (c_value - interval[index][0])/(interval[index][1] - interval[index][0])
        if (len(lst) >= 2 and len(lst) % 2 != 0) and (lst[-2] == 0 and lst[-1] == 0):
            break;
    return lst