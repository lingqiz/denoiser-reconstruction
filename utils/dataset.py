import os, cv2, torch, torchvision, scipy.io, numpy as np
import matplotlib.pylab as plt
from skimage.metrics import peak_signal_noise_ratio

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
def test_model(test_set, model, noise, device):
    model.eval()

    test_torch = torch.from_numpy(test_set).permute(0, 3, 1, 2).contiguous()
    test_noise = test_torch + torch.normal(0, noise / 255.0, size=test_torch.size())

    with torch.no_grad():
        residual = model(test_noise.to(device))
    
    noise_set = np.clip(test_noise.permute(0, 2, 3, 1).numpy(), 0, 1)
    denoise_set = np.clip((test_noise - 
        residual.detach().cpu()).permute(0, 2, 3, 1).numpy(), 0, 1)

    # calculate the PSNR for each test images
    psnr = np.zeros([2, test_torch.shape[0]])
    for idx, test, noisy, denoise in \
        zip(range(test_torch.shape[0]), test_set, noise_set, denoise_set):
        psnr[0, idx] = peak_signal_noise_ratio(test, noisy)
        psnr[1, idx] = peak_signal_noise_ratio(test, denoise)

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
        
        # load other dataset from files
        return DataFromFile(args, test_mode=test_mode)

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
                    'artwork' : ((64, 64), (64, 64), np.linspace(0.1, 1.0, 10), np.linspace(0.1, 1.0, 10))}

    def __init_para(self, args):        
        for idx, key in enumerate(self.DATASET_KEY):
            if getattr(args, key) is None:
                setattr(args, key, self.DATASET_PARA[args.data_path][idx])

    # read images under the specified directory
    def __init__(self, args, test_mode=False):
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

class MNIST(DataSet):
    def __init__(self):
        # load MNIST dataset
        train = torchvision.datasets.MNIST('./utils/dataset', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])).data
        test = torchvision.datasets.MNIST('./utils/dataset', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])).data
        mnist = torch.cat([train, test])

        # make them color images
        all_image = []
        for sample in mnist:
            sample = sample.numpy()
                
            image = np.empty([28, 28, 3])    
            image[sample == 0, :] = np.random.rand(3, )
            image[sample != 0, :] = np.random.rand(3, )

            all_image.append(image.astype(np.single))

        all_image = np.stack(all_image)
        np.random.shuffle(all_image)
        
        n_test = 500
        self.test_patches = all_image[:n_test, :]
        self.train_patches = all_image[n_test:, :]
