import matplotlib.pylab as plt
import os

class ISLVRC():
    """
    Load images from the ISLVRC dataset
    """

    def __init__(self):
        self.train_folder = './utilities/islvrc/train'
        self.train_images = []

        self.test_folder = './utilities/islvrc/test'
        self.test_images = []
        
        for file_name in os.listdir(self.train_folder):
            image = plt.imread(os.path.join(self.train_folder, file_name))
            image = image / 255.0
            if len(image.shape) == 3:
                self.train_images.append(image)
                
        for file_name in os.listdir(self.test_folder):
            image = plt.imread(os.path.join(self.test_folder, file_name))
            image = image / 255.0

            if len(image.shape) == 3:
                self.test_images.append(image)
        