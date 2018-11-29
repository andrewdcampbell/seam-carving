import numpy as np
import cv2
from numba import jit
from skimage import io, transform, util, filters, color
from scipy import ndimage as ndi

VIS_COLOR = np.array([255, 255, 255])
DEBUG = True
SHOULD_RESIZE = True
RESIZE_WIDTH = 1200
MASK_CONSTANT = 100000.0

def visualize(im, boolmask=None):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = VIS_COLOR
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def backward_energy_map(im):
    rgbx = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    rgby = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    rgbx = np.sum(rgbx**2, axis=2)
    rgby = np.sum(rgby**2, axis=2)
    
    return np.sqrt(rgbx + rgby)    

@jit
def forward_energy_map(im):
    """
    https://github.com/axu2/improved-seam-carving/blob/master/Improved%20Seam%20Carving.ipynb
    """
    h, w = im.shape[:2]
    # TODO this is cancer
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
        
    return energy

@jit
def add_seam(im, seam_idx):
    m, n = im.shape[:2]
    output = np.zeros((m, n + 1, 3))
    for row in range(m):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output

@jit
def add_seam_mask(im, seam_idx):
    m, n = im.shape[:2]
    output = np.zeros((m, n + 1))
    for row in range(m):
        col = seam_idx[row]
        if col == 0:
            p = np.average(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = np.average(im[row, col - 1: col + 1])
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]

    return output    



# Next two functions adapted from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

@jit
def remove_seam(im, boolmask, mask=None):
    h, w = im.shape[:2]

    boolmask3c = np.stack([boolmask] * 3, axis=2)
    im = im[boolmask3c].reshape((h, w - 1, 3))

    if mask is not None:
        mask = mask[boolmask].reshape((h, w - 1))
    return im, mask

@jit
def get_minimum_seam(im, mask=None, energyfn=forward_energy_map, removal=False):
    h, w = im.shape[:2]
    M = energyfn(im)

    # TODO: lets normalize the mask to be 0 or 1 first to make this cleaner
    # Maybe use an even larger positive value
    if mask is not None:
        M[np.where(mask > 10)] = -MASK_CONSTANT if removal else MASK_CONSTANT

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack path through DP matrix
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask


class SeamCarver:
    def __init__(self, filename, dy, dx, protect_mask='', object_mask=''):
        # initialize parameter
        self.filename = filename
        self.mask = None

        # read in image and store as np.float64 format
        self.in_image = cv2.imread(filename)
        h, w = self.in_image.shape[:2]
        if SHOULD_RESIZE and w > RESIZE_WIDTH:
            self.in_image = resize(self.in_image, width=RESIZE_WIDTH)

        self.in_image = self.in_image.astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        self.out_height, self.out_width = self.in_height, self.in_width 
        self.out_height += dy
        self.out_width += dx


        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        self.protect = not object_mask and protect_mask
        self.object = (object_mask != '')
        
        if self.protect or self.object:
            self.mask = cv2.imread(protect_mask, 0) if self.protect else cv2.imread(object_mask, 0)
            h, w = self.mask.shape[:2]
            if SHOULD_RESIZE and w > RESIZE_WIDTH:
                self.mask = resize(self.mask, width=RESIZE_WIDTH)      
            self.mask = self.mask.astype(np.float64)          

        self.start()


    def start(self):
        """
        :return:

        If object mask is provided --> object removal function will be executed
        else --> seam carving function (image retargeting) will be process
        """
        if self.object:
            self.object_removal()
        else:
            self.seams_carving()


    def seams_carving(self):
        """
        :return:

        We first process seam insertion or removal in vertical direction then followed by horizontal direction.

        If targeting height or width is greater than original ones --> seam insertion,
        else --> seam removal

        The algorithm is written for seam processing in vertical direction (column), so image is rotated 90 degree
        counter-clockwise for seam processing in horizontal direction (row)
        """

        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_image(self.mask, 1)
            self.seams_removal(delta_row * -1)
            self.out_image = self.rotate_image(self.out_image, 0)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            if self.protect:
                self.mask = self.rotate_image(self.mask, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, 0)


    def object_removal(self):
        """
        :return:

        Object covered by mask will be removed first and seam will be inserted to return to original image dimension
        """
        SHOULD_ROTATE = True #CHANGE ME
        rotate = False
        object_height, object_width = self.get_object_dimension()
        if SHOULD_ROTATE and object_height < object_width:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.mask = self.rotate_image(self.mask, 1)
            rotate = True

        while len(np.where(self.mask > 10)[0]) > 0:
            seam_idx, boolmask = get_minimum_seam(self.out_image, self.mask, removal=True)
            if DEBUG:
                visualize(self.out_image, boolmask)            
            self.out_image, self.mask = remove_seam(self.out_image, boolmask, self.mask)

        if not rotate:
            num_pixels = self.in_width - self.out_image.shape[1]
        else:
            num_pixels = self.in_height - self.out_image.shape[1]

        self.seams_insertion(num_pixels)
        if rotate:
            self.out_image = self.rotate_image(self.out_image, 0)


    def seams_removal(self, num_pixel):
        for _ in range(num_pixel):
            seam_idx, boolmask = get_minimum_seam(self.out_image, self.mask)
            if DEBUG:
                visualize(self.out_image, boolmask)            
            self.out_image, self.mask = remove_seam(self.out_image, boolmask, self.mask)


    def seams_insertion(self, num_pixel):
        seams_record = []
        temp_im = self.out_image.copy()
        temp_mask = self.mask.copy() if self.mask is not None else None

        for _ in range(num_pixel):
            seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask)
            if DEBUG:
                visualize(temp_im, boolmask)

            seams_record.append(seam_idx)
            temp_im, temp_mask = remove_seam(temp_im, boolmask, temp_mask)

        seams_record.reverse()

        for _ in range(len(seams_record)):
            seam = seams_record.pop()
            self.out_image = add_seam(self.out_image, seam)
            if DEBUG:
                visualize(self.out_image)
            if self.mask is not None:
                self.mask = add_seam_mask(self.mask, seam)
            seams_record = self.update_seams(seams_record, seam)
             


    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        res = b_energy + g_energy + r_energy
        return res


    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                output[row, col] = \
                    energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return output


    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
                 
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output


    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output


    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3), dtype=np.uint8)
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = output


    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output


    def rotate_image(self, image, clockwise):
        k = 1 if clockwise else 3
        return np.rot90(image, k)


    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, : ] = np.delete(self.mask[row, : ], [col])
        self.mask = np.copy(output)


    def add_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n + 1))
        for row in range(m):
            col = seam_idx[row]
            if col == 0:
                p = np.average(self.mask[row, col: col + 2])
                output[row, col] = self.mask[row, col]
                output[row, col + 1] = p
                output[row, col + 1: ] = self.mask[row, col: ]
            else:
                p = np.average(self.mask[row, col - 1: col + 1])
                output[row, : col] = self.mask[row, : col]
                output[row, col] = p
                output[row, col + 1: ] = self.mask[row, col: ]
        self.mask = np.copy(output)


    def get_object_dimension(self):
        rows, cols = np.where(self.mask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width


    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))



