import matplotlib.pylab as plt
import numpy as np
import os
import cv2

class ISLVRC():
    """
    Load images from the ISLVRC dataset
    """

    # sample image patches
    @staticmethod
    def sample_patch(image, scales, patch_size):
        samples = []
        for scale in scales:
            resized = cv2.resize(image, 
                    (int(image.shape[0] * scale), int(image.shape[1] * scale)))
            
            ih, iw = (resized.shape[0], resized.shape[1])
            ph, pw = patch_size

            if ih >= ph and iw >= pw:
                for x in range(0, ih - ph + 1, ph):
                    for y in range(0, iw - pw + 1, pw):
                        samples.append(resized[x:x+ph, y:y+pw, :])

        return samples

    # read images under the islvrc directory
    def __init__(self, args, linear=False):
        self.train_folder = './utilities/islvrc/train'
        self.test_folder = './utilities/islvrc/test'
        self.test_images = []
        self.linear = linear
                
        self.train_patches = []
        for file_name in os.listdir(self.train_folder):
            image = plt.imread(os.path.join(self.train_folder, file_name))
            if len(image.shape) == 3:
                image = self.to_float(image).astype(np.single)
                for sample in self.sample_patch(image, args.scales, args.patch_size):
                    self.train_patches.append(sample)                    

        self.train_patches = np.stack(self.train_patches)
        
        for file_name in os.listdir(self.test_folder):
            image = plt.imread(os.path.join(self.test_folder, file_name))
            if len(image.shape) == 3:
                image = self.to_float(image)
                self.test_images.append(image)

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
        return self.test_images
