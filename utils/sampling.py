import cv2, numpy as np
from tqdm.notebook import tqdm

def blur_subsample(image, blur_factor=1, sub_factor=1):
    """
    Blur and subsample an image.
    """
    blur = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
    sub_sample = blur[::sub_factor, ::sub_factor, :]

    return sub_sample

def forward_matrix(image_size, blur_factor, sub_factor, pbar=None):
    """
    Compute the forward matrix representing the blur and subsample operations.
    Note that we assume Column-major ordering for image -> vector (MATLAB style)
    """
    basis = np.zeros(image_size)
    result = blur_subsample(basis, blur_factor, sub_factor)

    render = np.zeros((result.size, np.prod(image_size)))

    close_pbar = False
    if pbar is None:
        close_pbar = True
        pbar = tqdm(total=np.prod(image_size))

    # indexing consistent with Column-Major order (MATLAB style)
    # loop over x -> y -> z
    # Python use Row-Major order (C style), z -> y -> x
    count = 0
    for idz in range(image_size[2]):
        for idy in range(image_size[1]):
            for idx in range(image_size[0]):
                basis = np.zeros(image_size)
                basis[idx, idy, idz] = 1.0

                result = blur_subsample(basis, blur_factor, sub_factor)
                render[:, count] = result.flatten()

                count += 1
                pbar.update(1)

    # close pbar if opened within function
    if close_pbar:
        pbar.close()

    return render