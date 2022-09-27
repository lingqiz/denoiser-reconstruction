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
    index = np.zeros(im_size)

    if mode == 'full':
        factor = paras['factor']
        index[::factor, ::factor, :] = 1

        return index.astype(np.bool)

    if mode == 'regular':
        factor = paras['factor']

        # assign RGB plane separately
        for pid in range(3):
            index[pid::factor, pid::factor, pid] = 1

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