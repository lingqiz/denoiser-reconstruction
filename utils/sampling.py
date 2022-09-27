import cv2, numpy as np
from tqdm.notebook import tqdm

def gauss_blur(image, factor=1):
    """
    Adding Gaussian blur to an image
    """
    return cv2.GaussianBlur(image, (factor, factor), 0)

def sample_mtx(im_size, mode, paras):
    """
    Create a matrix that represents the subsample operation applied to an image
    """
    R = 0; G = 1; B = 2
    index = np.zeros(im_size)

    if mode == 'full':
        # full RGB space at each spatial location
        factor = paras['factor']
        index[::factor, ::factor, :] = 1

        return index.astype(np.bool)

    if mode == 'regular':
        # assign RGB plane separately
        # regular bayer-like pattern
        factor = paras['factor']

        index[0::factor, 0::factor, R] = 1
        index[1::factor, 1::factor, B] = 1

        index[0::factor, 1::factor, G] = 1
        index[1::factor, 0::factor, G] = 1

        return index.astype(np.bool)

    if mode == 'random':
        # randomly assign location of R, G and B sample
        ratio = paras['ratio']
        n_pix = im_size[0] * im_size[1]
        n_smp = int(n_pix * ratio)

        # loop through RGB plane
        for idx in range(3):
            sample = np.zeros(n_pix)
            sample[:n_smp] = 1
            np.random.shuffle(sample)

            index[:, :, idx] = np.reshape(sample, im_size[:2])

        return index.astype(np.bool)

    if mode == 'exclusive':
        # cone mosaic where the total RGB sample is fixed
        n_pix = im_size[0] * im_size[1]
        assign = (paras['assign'] * paras['ratio'] * n_pix).astype(np.int)

        # add the correct numner of samples
        array = [0] * assign[0] + [1] * assign[1] + [2] * assign[2]
        array += [3] * (n_pix - len(array))
        np.random.shuffle(array)

        # assign to the subsample image
        array = np.reshape(array, im_size[:2])
        for idx in range(3):
            index[:, :, idx] = (array == idx)

        return index.astype(np.bool)

def forward_mtx(im_size, blur, smp_mtx, pbar=None):
    """
    Compute the forward matrix representing the blur and subsample operations.
    Note that we assume Column-major ordering for image -> vector (MATLAB style)
    """
    basis = np.zeros(im_size)
    result = gauss_blur(basis, blur)[smp_mtx]

    render = np.zeros((result.size, np.prod(im_size)))

    close_pbar = False
    if pbar is None:
        close_pbar = True
        pbar = tqdm(total=np.prod(im_size))

    # indexing consistent with Column-Major order (MATLAB style)
    # loop over x -> y -> z
    # Python use Row-Major order (C style), z -> y -> x
    count = 0
    for idz in range(im_size[2]):
        for idy in range(im_size[1]):
            for idx in range(im_size[0]):
                basis = np.zeros(im_size)
                basis[idx, idy, idz] = 1.0

                result = gauss_blur(basis, blur)[smp_mtx]
                render[:, count] = result.flatten()

                count += 1
                pbar.update(1)

    # close pbar if opened within function
    if close_pbar:
        pbar.close()

    return render

def forward_matrix(image_size, blur_factor, sub_factor, pbar=None):
    """
    (legacy function that assumes a full RGB sample at each spatial location)
    Compute the forward matrix representing the blur and subsample operations.
    Note that we assume Column-major ordering for image -> vector (MATLAB style)
    """
    smp_mtx = sample_mtx(image_size, mode='full',
                        paras={'factor', sub_factor})

    return forward_mtx(image_size, blur_factor, smp_mtx, pbar)