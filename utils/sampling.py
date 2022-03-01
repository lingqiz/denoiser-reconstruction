import cv2, numpy as np
from tqdm.notebook import tqdm

def blur_subsample(image, blur_factor=1, sub_factor=1):
    """
    Blur and subsample an image.
    """
    blur = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
    sub_sample = blur[::sub_factor, ::sub_factor, :]

    return sub_sample


def forward_matrix(image_size, blur_factor, sub_factor):
    """
    Compute the forward matrix representing the blur and subsample operations.
    Note that we assume Column-major ordering for image -> vector (MATLAB style)
    """
    basis = np.zeros(image_size)
    result = blur_subsample(basis, blur_factor, sub_factor)

    render = np.zeros((result.size, np.prod(image_size)))
    pbar = tqdm(total=np.prod(image_size))

    # indexing consistent with Column-Major order (MATLAB style)
    count = 0
    for idz in range(image_size[2]):
        for idx in range(image_size[0]):
            for idy in range(image_size[1]):
                basis = np.zeros(image_size)
                basis[idx, idy, idz] = 1.0

                result = blur_subsample(basis, blur_factor, sub_factor)
                render[:, count] = np.transpose(result, [2, 1, 0]).flatten()

                count += 1
                pbar.update(1)

    pbar.close()
    return render