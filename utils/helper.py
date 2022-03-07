import h5py, torch, numpy as np, matplotlib.pyplot as plt
from utils.dataset import test_model, gamma_correct
from inverse.sampler import sample_prior
from tqdm import tqdm

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

def compute_svd(msmt_mtx):
    u, s, _ = np.linalg.svd(msmt_mtx.T, full_matrices=False)
    return (u.T, s ** 2)