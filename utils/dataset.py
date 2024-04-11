import os, cv2, torch, torchvision, scipy.io, numpy as np
import matplotlib.pylab as plt
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

# sample image patches
def sample_patch(image, scales, patch_size):
    samples = []
    for scale in scales:
        # cv2 resize (width, height)
        resized = cv2.resize(image,
                (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        ih, iw = (resized.shape[0], resized.shape[1])
        ph, pw = patch_size

        if ih >= ph and iw >= pw:
            for x in range(0, ih - ph + 1, ph):
                for y in range(0, iw - pw + 1, pw):
                    samples.append(resized[x:x+ph, y:y+pw, :])

    return samples

# image gamma linear and correction
GAMMA_TABLE = scipy.io.loadmat('./assets/gamma.mat')['gammaTable']

def gamma_linear(image):
    image_linear = np.zeros(image.shape)
    for idx in range(image.shape[-1]):
        image_linear[:, :, idx] = np.interp(image[:, :, idx],
            np.linspace(0.0, 1.0, GAMMA_TABLE.shape[0]), GAMMA_TABLE[:, idx])

    return image_linear

def gamma_correct(image):
    image_correct = np.zeros(image.shape)
    for idx in range(image.shape[-1]):
        image_correct[:, :, idx] = np.interp(image[:, :, idx],
            GAMMA_TABLE[:, idx], np.linspace(0.0, 1.0, GAMMA_TABLE.shape[0]))

    return image_correct

# test image denoising model
def test_model(test_set, model, noise, device, data_range=None, clip_range=(0, 1)):
    model.eval()

    # make sure things are the right data type
    if type(test_set) is torch.Tensor:
        test_torch = test_set.cpu()
        test_set = test_set.numpy()
    else:
        test_torch = torch.from_numpy(test_set)

    test_torch = test_torch.permute(0, 3, 1, 2).contiguous()
    test_noise = test_torch + torch.normal(0, noise / 255.0, size=test_torch.size())

    with torch.no_grad():
        residual = model(test_noise.to(device))

    noise_set = np.clip(test_noise.permute(0, 2, 3, 1).numpy(),
                        clip_range[0], clip_range[1])

    denoise_set = np.clip((test_noise - residual.detach().cpu()).permute(0, 2, 3, 1).numpy(),
                        clip_range[0], clip_range[1])

    # calculate the PSNR for each test images
    psnr = np.zeros([2, test_torch.shape[0]])
    for idx, test, noisy, denoise in \
        zip(range(test_torch.shape[0]), test_set, noise_set, denoise_set):
        psnr[0, idx] = peak_signal_noise_ratio(test, noisy, data_range=data_range)
        psnr[1, idx] = peak_signal_noise_ratio(test, denoise, data_range=data_range)

    return (psnr, test_set, noise_set, denoise_set)

class DataSet:
    """
    Base class for training/testing dataset
    """
    @staticmethod
    def load_dataset(args, test_mode=False):
        # mnist is loaded directly through torchvision
        if args.data_path == 'mnist':
            return MNIST()
        if args.data_path.startswith('npy_'):
            return NPYImage(args.data_path)

        # load other dataset from files
        return DataFromFile(args, test_mode)

    # return sampled images from the training set
    def train_set(self):
        return self.train_patches

    # return a python array of images in the test set
    def test_set(self):
        return self.test_patches

class DataFromFile(DataSet):
    DATASET_KEY  = ['patch_size', 'test_size', 'scales', 'test_scale']
    DATASET_PARA = {'islvrc' : ((48, 48), (128, 128), [1.0, 0.80, 0.60, 0.40, 0.20], [0.5]),
                    'lfw' : ((128, 128), (128, 128), [128.0 / 250.0], [128.0 / 250.0]),
                    'celeba' : ((50, 40), (50, 40), [50.0 / 218.0], [50.0 / 218.0]),
                    'artwork' : ((128, 128), (128, 128), \
                                 np.linspace(0.5, 1.0, 4), np.linspace(0.5, 1.0, 4))}

    def __init_para(self, args):
        for idx, key in enumerate(self.DATASET_KEY):
            if getattr(args, key) is None:
                setattr(args, key, self.DATASET_PARA[args.data_path][idx])

    # read images under the specified directory
    def __init__(self, args, test_mode=False):
        # load the resized celeba dataset from npy file
        if args.data_path == 'celeba_resize':
            npy_path = os.path.join('utils', 'dataset', 'celeba', 'celeba_resize.npy')
            with open(npy_path, 'rb') as fl:
                self.train_patches = np.load(fl)
                self.test_patches = np.load(fl)
            return

        # load other dataset from file
        self.__init_para(args)

        self.train_folder = os.path.join('utils', 'dataset', args.data_path, 'train')
        self.test_folder = os.path.join('utils', 'dataset', args.data_path, 'test')

        self.test_images = []
        self.linear = args.linear

        # sample individual patches for training
        self.train_patches = []
        if not test_mode:
            for file_name in os.listdir(self.train_folder):
                image = plt.imread(os.path.join(self.train_folder, file_name))
                if len(image.shape) == 3:
                    for sample in sample_patch(image, args.scales, args.patch_size):
                        sample = self.to_float(sample).astype(np.single)
                        if sample.mean() >= 0.05:
                            self.train_patches.append(sample)
            # training set
            self.train_patches = np.stack(self.train_patches)

        # sample for testing set
        test_size = args.test_size
        test_scale = args.test_scale
        self.test_patches = []
        for file_name in os.listdir(self.test_folder):
            image = plt.imread(os.path.join(self.test_folder, file_name))
            if len(image.shape) == 3:
                image = self.to_float(image).astype(np.single)
                for sample in sample_patch(image, test_scale, test_size):
                    self.test_patches.append(sample)
        # test set
        self.test_patches = np.stack(self.test_patches)

    # convert to float point images
    def to_float(self, image):
        if self.linear:
            return gamma_linear(image / 255.0)
        else:
            return image / 255.0

class CelebA(DataSet):
    def __init__(self, from_numpy=False):
        # load the resized celeba dataset from npy file
        if from_numpy:
            npy_path = os.path.join('utils', 'dataset', 'celeba', 'celeba_gray.npy')
            with open(npy_path, 'rb') as fl:
                self.train_patches = np.expand_dims(np.load(fl), axis=3)
                self.test_patches = np.expand_dims(np.load(fl), axis=3)
            return

        # read images from file
        self.train_folder = os.path.join('utils', 'dataset', 'celeba', 'train')
        self.test_folder = os.path.join('utils', 'dataset', 'celeba', 'test')

        # sample individual patches for training
        self.train_patches = []
        for file_name in tqdm(os.listdir(self.train_folder)):
            image = plt.imread(os.path.join(self.train_folder, file_name))

            if len(image.shape) == 3:
                image = image / 255.0
                image = image[29:-29, 9:-9] # crop to 160x160
                image = cv2.resize(image, (80, 80),
                                   interpolation = cv2.INTER_AREA)
                self.train_patches.append(image)

        self.train_patches = np.stack(self.train_patches).astype(np.single)
        self.train_patches = np.mean(np.stack(self.train_patches), axis=3)

        N_TEST = 1000
        self.test_patches = np.expand_dims(self.train_patches[0:N_TEST, :], axis=3)
        self.train_patches = np.expand_dims(self.train_patches[N_TEST:, :], axis=3)

        return

class Texture(DataSet):
    def __init__(self, class_name='all'):
        # load the resized celeba dataset from npy file
        npy_path = os.path.join('utils', 'dataset', 'texture', 'texture_%s.npy')
        npy_path = npy_path % class_name

        with open(npy_path, 'rb') as fl:
            self.train_patches = np.load(fl)
            self.test_patches = np.load(fl)

        return

class CIFAR(DataSet):
    def __init__(self):
        npy_path = os.path.join('utils', 'dataset', 'cifar10', 'cifar10.npy')
        with open(npy_path, 'rb') as fl:
            self.train_patches = np.load(fl)
            self.test_patches = np.load(fl)

        return

class MNIST(DataSet):
    def __init__(self):
        # load MNIST dataset
        train = torchvision.datasets.MNIST('./utils/dataset', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))

        test = torchvision.datasets.MNIST('./utils/dataset', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))

        mnist = torch.cat([train.data, test.data])
        target = torch.cat([train.targets, test.targets]).numpy()

        # reshape and normalize
        all_image = []
        for sample in mnist:
            sample = sample.numpy() / 255.0
            image = np.reshape(sample, [28, 28, 1])
            all_image.append(image.astype(np.single))

        all_image = np.stack(all_image)

        # random shuffle
        rng = np.random.default_rng(seed=0)
        rng.shuffle(all_image, axis=0)

        rng = np.random.default_rng(seed=0)
        rng.shuffle(target, axis=0)

        # split into training and testing set
        n_test = 512
        self.test_patches = all_image[:n_test, :]
        self.train_patches = all_image[n_test:, :]

        self.test_label = target[:n_test]
        self.train_label = target[n_test:]

    def select_digit(self, digit):
        # select only a single digit
        self.train_patches = self.train_patches[self.train_label == digit]
        self.test_patches = self.test_patches[self.test_label == digit]

        return

class NPYImage(DataSet):
    def __init__(self, data_path):
        '''
        Load image dataset with npy format, options are:
            - npy_cifar_all
            - npy_cifar_10
            - npy_cifar_cars
            - npy_celeba_tiny
        '''
        # file path for CIFAR dataset
        file_path = os.path.join('utils', 'dataset', 'NPYs',
                                 data_path[4:] + '.npy')

        # load npy file
        with open(file_path, 'rb') as fl:
            test = np.load(fl)
            train = np.load(fl)

        self.test_patches = test.astype(np.float32)
        self.train_patches = train.astype(np.float32)

class Mixture(DataSet):
    def __init__(self):
        # load the mixture dataset from npy file
        npy_path = os.path.join('utils', 'dataset', 'celeba_bedroom_mixture.npy')
        with open(npy_path, 'rb') as fl:
            train = np.load(fl)
            test = np.load(fl)

            self.train_index = np.load(fl)
            self.test_index = np.load(fl)

        self.train_patches = train.astype(np.float32)
        self.test_patches = test.astype(np.float32)