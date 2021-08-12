import torch, numpy as np
import PIL, PIL.ImageDraw, PIL.ImageFont
from abc import ABC, abstractmethod

# make letter stimulus
FONT_PATH = './assets/arial.ttf'
BACKGROUND = 0.25
IMG_MAX = 255.0

def letter_image(ltr, ltr_size, im_size):
    loc = (np.array(im_size) - ltr_size) / 2

    # draw letter
    image = PIL.Image.new('RGB', im_size, color=0)
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype(FONT_PATH, ltr_size)
    draw.text(loc, ltr, font=font)

    # find the window of the letter
    index = np.zeros(4)
    all_index = np.nonzero(image)

    index[0], index[1] = np.min(all_index[0]), np.max(all_index[0] + 1)
    index[2], index[3] = np.min(all_index[1]), np.max(all_index[1] + 1)

    # add background and make image / torch stimulus
    image = np.clip(np.array(image) / IMG_MAX + BACKGROUND, 0, 1)
    stim = torch.tensor(image.astype(np.float32)).permute([2, 0, 1])

    return (image, stim, index.astype(np.int32))

class LetterTask(ABC):
    def __init__(self, ltr, ltr_size, im_size):
        self.ltr = ltr
        self.tid = ord(ltr) - ord('A')

        # setup an array of templates
        self.templates, self.indices = self._make_templates(ltr_size, im_size)

        # generate stimulus
        self.stimulus = self._make_stimulus(ltr_size, im_size)

    @abstractmethod
    def _make_stimulus(self, ltr_size, im_size):
        pass

    def _make_templates(self, ltr_size, im_size):
        # produce a set of templates
        templates = []
        indices = []
        for ltr in map(chr, range(ord('A'), ord('Z')+1)):
            # create letter image
            ltr_image, _, index = letter_image(ltr, ltr_size=ltr_size, im_size=im_size)
            ltr_image = ltr_image[index[0]:index[1], index[2]:index[3], :]

            # normlized template
            templates.append(ltr_image.flatten() / np.linalg.norm(ltr_image.flatten()))
            indices.append(index)

        return templates, indices

    def compute_dv(self, x):
        # simulate observer, compute decision variable
        # (as dot product of template and signal)
        dv = []
        for tpl, idx in zip(self.templates, self.indices):
            signal = x[idx[0]:idx[1], idx[2]:idx[3], :]
            signal = signal.flatten() / np.linalg.norm(signal.flatten())
            dv.append(np.dot(tpl, signal))

        dv = np.array(dv)

        mask = np.ones_like(dv)
        mask[self.tid] = 0

        return dv, dv[self.tid] - np.mean(dv[mask.astype(bool)])

    def eval_model(self, model):
        recon = model(self.stimulus)
        return self.compute_dv(recon)

class LetterDetection(LetterTask):
    def _make_stimulus(self, ltr_size, im_size):
        return letter_image(self.ltr, ltr_size=ltr_size, im_size=im_size)[1]