import argparse, cupy as cp
import h5py, torch, numpy as np, matplotlib.pyplot as plt
from utils.dataset import test_model, gamma_correct
from inverse.sampler import sample_prior
from tqdm import tqdm

# argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Denoiser Training')

    # option/mode for the script
    parser.add_argument('-f',
                        required=False,
                        type=str,
                        help='jupyter notebook')
    parser.add_argument('--mode',
                        required=False,
                        type=str,
                        help='script mode')
    parser.add_argument('--model_path',
                        type=str,
                        default='./assets/conv3_ln.pt')

    # arguments for network training
    parser.add_argument('--batch_size',
                        type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=100,
                        help='number of epochs to train')
    parser.add_argument('--noise_level',
                        default=[0, 200])
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)    
    parser.add_argument('--decay_lr',
                        type=float,
                        default=0.98)
    parser.add_argument('--decay_adam',
                        type=float,
                        default=0.1)    
    parser.add_argument('--bias_sd',
                        type=bool,
                        default=False)
    parser.add_argument('--scale_image',
                        type=bool,
                        default=False)
    parser.add_argument('--opt_index',
                        type=int,
                        default=0)
    parser.add_argument('--ddp',
                        type=bool,
                        default=False,
                        help='Distributed Data Parallel')
    parser.add_argument('--save_path',
                        type=str,
                        default='./assets/model_para.pt')

    # see dataset.py for parameters for individual dataset
    parser.add_argument('--data_path',
                        type=str,
                        default='islvrc')
    parser.add_argument('--linear',
                        type=bool,
                        default=True)
    parser.add_argument('--patch_size',
                        default=None)
    parser.add_argument('--test_size',
                        default=None)
    parser.add_argument('--scales',
                        default=None)
    parser.add_argument('--test_scale',
                        default=None)

    # network architecture
    parser.add_argument('--padding',
                        type=int,
                        default=1)
    parser.add_argument('--kernel_size',
                        type=int,
                        default=3)
    parser.add_argument('--num_kernels',
                        type=int,
                        default=64)
    parser.add_argument('--num_layers',
                        type=int,
                        default=20)
    parser.add_argument('--im_channels',
                        type=int,
                        default=3)
    parser.add_argument('--save_model',
                        type=bool,
                        default=True)

    # parse arguments and check
    args, _ = parser.parse_known_args()
    return args

# denoiser demo
def plot_denoiser(test, model, noise, n_plot, device='cpu', gamma=True):
    result = test_model(test, model, noise=noise, device=device)

    psnr = np.mean(result[0], axis=1)
    print('psnr in: %.2f, out: %.2f' % (psnr[0], psnr[1]))

    sample_idx = np.random.choice(range(test.shape[0]),
                                  size=n_plot, replace=False)

    fig, axs = plt.subplots(3, n_plot, figsize=(3 * n_plot, 9))
    for idx in range(n_plot):
        img_idx = sample_idx[idx]
        for idy in range(3):
            image = gamma_correct(result[idy + 1][img_idx]) if gamma else \
                    result[idy + 1][img_idx]
            axs[idy][idx].imshow(image)
            axs[idy][idx].axis('off')

    fig.tight_layout()
    return fig

# simple evaluation of the denoiser
def eval_denoiser(test, model, device='cpu'):
    # range of noise for testing
    noise_level = range(15, 110, 10)

    psnr_in = np.zeros([len(noise_level), 1])
    psnr_out = np.zeros([len(noise_level), test.shape[0]])

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

# sample from a prior
def plot_sample(model, beta, im_size, n_sample=25, mu=0.25, gamma=True):
    samples = []
    for idx in tqdm(range(n_sample)):
        init = mu + torch.randn(size=im_size)
        samples.append(sample_prior(model, init, beta=beta, stride=0)[-1])

    edge = int(np.ceil(np.sqrt(n_sample)))
    fig, axs = plt.subplots(edge, edge, figsize=(12, 12), sharex=True, sharey=True)
    for idx, ax in zip(range(n_sample), axs.flat):
        image = np.clip(samples[idx], 0, 1)
        image = gamma_correct(image) if gamma else image

        ax.imshow(image)
        ax.axis('off')

    fig.tight_layout()
    return fig

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

# compute a SVD for the measurement matrix
def compute_svd(msmt_mtx):
    # create a CP matrix
    mtx = cp.array(msmt_mtx.astype(np.float32))
    # SVD on GPU
    u, s, _ = cp.linalg.svd(mtx.T, full_matrices=False)
    return (cp.asnumpy(u.T), cp.asnumpy(s))