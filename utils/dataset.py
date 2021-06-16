import os, cv2, numpy as np
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

def test_model(test_set, model, noise, device):
    test_torch = torch.from_numpy(test_set).permute(0, 3, 1, 2).contiguous()
    test_noise = test_torch + torch.normal(0, noise / 255.0, test_torch.size())

    with torch.no_grad():
        residual = model(test_noise.to(device))
    
    denoised = np.clip((test_noise - 
        residual.detach().cpu()).permute(0, 2, 3, 1).numpy(), 0, 1)

    
    

class ISLVRC():
    """
    Load images from the ISLVRC dataset
    """

    # read images under the islvrc directory
    def __init__(self, args, linear=False):
        self.train_folder = './utils/islvrc/train'
        self.test_folder = './utils/islvrc/test'
        self.test_images = []
        self.linear = linear
        
        # sample individual patches
        self.train_patches = []
        for file_name in os.listdir(self.train_folder):
            image = plt.imread(os.path.join(self.train_folder, file_name))
            if len(image.shape) == 3:
                image = self.to_float(image).astype(np.single)
                for sample in sample_patch(image, args.scales, args.patch_size):
                    if sample.mean() >= 0.05:
                        self.train_patches.append(sample)

        # training set
        self.train_patches = np.stack(self.train_patches)
        
        # testing set
        test_size = (128, 128)
        test_scale = [0.5]
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
            # linearization with display gamma table
            pass
        else:
            return image / 255.0

    # return sampled images from the training set
    def train_set(self):
        return self.train_patches

    # return a python array of images in the test set
    def test_set(self):
        return self.test_patches
