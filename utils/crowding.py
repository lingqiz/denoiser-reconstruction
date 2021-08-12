import torch, numpy as np
import PIL, PIL.ImageDraw, PIL.ImageFont
from abc import ABC

# make letter stimulus
FONT_PATH = './assets/arial.ttf'
BACKGROUND = 0.25
IMG_MAX = 255.0

def letter_image(ltr, ltr_size, im_size):
    image = PIL.Image.new('RGB', im_size, color=0)
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype(FONT_PATH, ltr_size)

    # draw letter
    loc = (np.array(im_size) - ltr_size) / 2
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

def letter_array(letters, ltr_size, gap_size, im_size):
    image = PIL.Image.new('RGB', im_size, color=0)
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype('./assets/arial.ttf', ltr_size)

    loc = (im_size[0] - ltr_size) / 2
    draw.text((loc - ltr_size / 2 - gap_size, loc), letters[0], font=font)
    draw.text((loc, loc), letters[1], font=font)
    draw.text((loc + ltr_size / 2 + gap_size, loc), letters[2], font=font)

    # add background and make image / torch stimulus
    image = np.clip(np.array(image) / IMG_MAX + BACKGROUND, 0, 1)
    stim = torch.tensor(image.astype(np.float32)).permute([2, 0, 1])

    return image, stim

# base class for setting up an array of templates
class LetterTask(ABC):
    def __init__(self, ltr, ltr_size, im_size):
        self.ltr = ltr
        self.tid = ord(ltr) - ord('A')

        # setup an array of templates
        self.templates, self.indices = self._init_templates(ltr_size, im_size)

    def _init_templates(self, ltr_size, im_size):
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

# class for single letter detection
class LetterDetection(LetterTask):
    def __init__(self, ltr, ltr_size, im_size):
        super().__init__(ltr, ltr_size, im_size)

        # generate a (static) stimulus
        self.stimulus = self._init_stimulus(ltr_size, im_size)
        self.stim_image = self.stimulus.detach().permute(1, 2, 0).numpy()

    def _init_stimulus(self, ltr_size, im_size):
        return letter_image(self.ltr, ltr_size=ltr_size, im_size=im_size)[1]

# crowding stimulus
class LetterCrowding(LetterTask):
    def __init__(self, ltr, ltr_size, im_size, gap_size=None, flank=None):
        super().__init__(ltr, ltr_size, im_size)

        self.ltr_size = ltr_size
        self.im_size = im_size

        if not ((gap_size is None) or (flank is None)):
            self.set_stimulus(gap_size, flank)

    def set_stimulus(self, gap_size, flank):
        letters = (flank[0], self.ltr, flank[1])

        self.stim_image, self.stimulus = \
            letter_array(letters, self.ltr_size, gap_size, self.im_size)
