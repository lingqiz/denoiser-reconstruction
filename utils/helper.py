import h5py
import numpy as np

# simple evaluation of the denoiser
def eval_denoiser(test, model, device='cpu'):
    # range of noise for testing
    noise_level = range(15, 110, 10)

    psnr_in = np.zeros([len(noise_level), 1])
    psnr_out = np.zeros([len(noise_level), test.shape()[0]])
    
    sd_true = np.zeros(len(noise_level))
    sd_est  = np.zeros(len(noise_level))
    
    # run denoising on the test set
    for idx, noise in enumerate(noise_level):
        psnr, test, noise, denoise = test_model(test, model, noise, device)

        psnr_in[idx] = psnr[0, ].mean()
        psnr_out[idx, ] = psnr[1, ]
        
        sd_true[idx] = np.std(noise - test)
        sd_est[idx]  = np.std(denoise - noise)
        
    return (psnr_in, psnr_out, sd_true, sd_est)

# read render array into numpy format
def read_array(file_path):
    data = h5py.File(file_path, 'r')
    
    img_size = np.array(data['imageSize'])
    ecc_x = np.array(data['eccX'])
    ecc_y = np.array(data['eccY'])
    
    ny, nx = data['renderArray'].shape
    
    # init array
    array = [[0 for y in range(ny)] 
             for x in range(nx)]
    
    # read matrices from data
    for x in range(nx):
        for y in range(ny):
            array[x][y] = np.array(data[data['renderArray'][y][x]], 
                                   dtype=np.single)
            
    return (array, img_size, (nx, ny), (ecc_x, ecc_y))