"""
Compress utilities file
"""
import numpy as np				# Import to process numpy array
import scipy.fftpack as sf		# Import to process 2-dimension dct

# Default transform matrix from RGB to YCbCr
yc_mat = np.asmatrix([[0.299, -0.1687, 0.5], [0.587, -0.3313, -0.4187], [0.114, 0.5, -0.0813]])
yc_pad = np.asmatrix([0, 128, 128])

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

def compress(block):
    """ Compress image data block function
    :param
        block: an 8x8 block with each item is [R, G, B]
    :return (Arithmetic coding) code for block
    """

    # Transform RGB to YCbCr
    yc_bl = np.zeros((8, 8, 3), dtype=np.int8)
    
    for i in range(8):
        for j in range(8):
            rgb_cmp = np.asmatrix(block[i][j])
            y,cb,cr = (np.array((rgb_cmp*yc_mat+yc_pad).astype(np.uint8))[0]-128).astype(np.int8)
            yc_bl[i][j] = np.array([y, cb, cr])
            
    # Switch YCbCr block to 3 block for each Y, Cb, Cr component and calculate DCT for them
    y_dct = sf.dct(yc_bl[:,:,0], norm='ortho')
    cb_dct = sf.dct(yc_bl[:,:,1], norm='ortho')
    cr_dct = sf.dct(yc_bl[:,:,2], norm='ortho')
    
    # From DCT data to quantization data
    y_quant = np.round(y_dct / quant_tbl).astype(np.int8)
    cb_quant = np.round(cb_dct / quant_tbl).astype(np.int8)
    cr_quant = np.round(cr_dct / quant_tbl)).astype(np.int8)
    
    # Convert 8x8 block to zigzag 1x64 block
    y_zz = zig_zag(y_quant)
    cb_zz = zig_zag(cb_quant)
    cr_zz = zig_zag(cr_quant)
    
    # Calc DC and AC, put together to list
    y_cmp, cb_cmp, cr_cmp = dc_and_ac_calc(y_zz, cb_zz, cr_zz)
    
    # Encode using entropy coding
    y_encode = encode(y_cmp)
    cb_encode = encode(cb_cmp)
    cr_encode = encode(cr_cmp)
    
    return [y_encode, cb_encode, cr_encode]


def zig_zag(b):
    """ Zig-zag transform function
    :param
        b: An 8x8 block
    :return An 1x64 block which transform from 8x8 block use zig-zag algorithm
    """
    return np.array([b[0,0], b[0,1], b[1,0], b[2,0], b[1,1], b[0,2], b[0,3], b[1,2], \
                     b[2,1], b[3,0], b[4,0], b[3,1], b[2,2], b[1,3], b[0,4], b[0,5], \
                     b[1,4], b[2,3], b[3,2], b[4,1], b[5,0], b[6,0], b[5,1], b[4,2], \
                     b[3,3], b[2,4], b[1,5], b[0,6], b[0,7], b[1,6], b[2,5], b[3,4], \
                     b[4,3], b[5,2], b[6,1], b[7,0], b[7,1], b[6,2], b[5,3], b[4,4], \
                     b[3,5], b[2,6], b[1,7], b[2,7], b[3,6], b[4,5], b[5,4], b[6,3], \
                     b[7,2], b[7,3], b[6,4], b[5,5], b[4,6], b[3,7], b[4,7], b[5,6], \
                     b[6,5], b[7,4], b[7,5], b[6,6], b[5,7], b[6,7], b[7,6], b[7,7]], dtype=np.int8)


def rle(ac):
    """ Run-length code algorithm
    :param
        ac: list of ac component of zig-zag output
    :return array of run-length code
    """

    # Count number of zero before a non-zero value
    count_zero = 0
    
    # A list which store run length code
    lst = []
    
    for i in range(len(ac)):
        if ac[i] == 0:
            count_zero += 1						# Increase count_zero if current value is zero
        else:
            lst.append([count_zero, ac[i]])		# Append count_zero and current non-zero value
            count_zero = 0						# Reset count_zero
    lst.append([0, 0])							# Ending run-length code is [0, 0]
    return np.array(lst)


def dc_and_ac_calc(y_cmp, cb_cmp, cr_cmp):
    """ Calculate dc and ac value
    :param
        y_cmp: array of Y component
        cb_cmp: array of Cb component
        cr_cmp: array of Cr component
    :return
        Three array: Y, Cb, Cr, each array 
        include DC and AC component continous
    """

    global y_pre_dc, cb_pre_dc, cr_pre_dc
    y_dc, cb_dc, cr_dc = y_cmp[0] - y_pre_dc, cb_cmp[0] - cb_pre_dc, cr_cmp[0] - cr_pre_dc
    y_pre_dc, cb_pre_dc, cr_pre_dc = y_cmp[0], cb_cmp[0], cr_cmp[0]
    
    y_ac, cb_ac, cr_ac = rle(y_cmp[1:]), rle(cb_cmp[1:]), rle(cr_cmp[1:])
    return np.append(y_dc, y_ac), np.append(cb_dc, cb_ac), np.append(cr_dc, cr_ac)


def encode(cmp):
    """ Encode DC and AC component to a float values 
        using Arithmetic coding encode algorithm
    :param
        cmp: An array of DC and AC component
    :return Encode value
    """
    
    low = np.float128(0)
    high = np.float128(1)
    range = high - low
    for s in cmp:
        if not s in interval:					# Round value in range [-32, 31]
            if s > 31:
                s = 31
            if s < -32:
                s = -32
        high = low + range * interval[s][1]		# Update high value
        low = low + range * interval[s][0]		# Update low value
        range = high - low						# Update range
    return low + (high - low) / 2
