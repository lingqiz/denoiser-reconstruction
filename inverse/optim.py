import torch, numpy as np, matplotlib.pyplot as plt
from torch.nn import MSELoss
from utils.dataset import gamma_correct
from inverse.solver import linear_inverse
from plenoptic.synthesize.eigendistortion import Eigendistortion

# constants associated with the function
MSE = MSELoss(reduction='mean')
IDENTITY = lambda x: x
CLAMP= lambda x: x.clamp_(0, 1)

# difference maximization
def max_diff(model, render, init, n_iter, opt_norm=0.01, stride=0,
            h_init=0.25, beta=0.25, sig_end=0.01, iter_tol=30, t_max=35,
            distance=MSE, generator=IDENTITY, constraint=CLAMP):

    if not isinstance(render, list):
        render = [render]

    seed = np.random.randint(0, 2**32)
    sequence = []
    for n in range(int(n_iter)):
        # clear gradient
        init.grad = None
        image_in = generator(init)

        # compute reconstruction for all matrices
        recon_list = []
        for mtx in render:
            recon, t, _ = linear_inverse(model, mtx, image_in,
                            h_init=h_init, beta=beta, sig_end=sig_end,
                            stride=0, seed=seed, t_max=t_max, with_grad=True)

            recon_list.append(recon.unsqueeze(0))

        # compute the distance between reconstruction and input
        recon = torch.cat(recon_list, dim=0)
        loss = distance(recon, image_in.unsqueeze(0)\
                    .repeat([len(recon_list), 1, 1, 1]))

        loss.backward()
        grad_norm = torch.norm(init.grad)

        if stride != 0 and n % stride == 0:
            sequence.append(init.detach().cpu())
            print('iter: %d, n_step: %d, norm: %.4f, loss: %.4f' %
                    (n, t, grad_norm, loss.item()))

        # increase the distance
        with torch.no_grad():
            init += opt_norm / grad_norm * init.grad
            constraint(init)

        # reduce interation length if needed
        if t >= iter_tol:
            beta *= 1.25
            sig_end *= 2.0

    return init, recon.squeeze(0), sequence

# eigendistortion
def eig_distort(model, input_set, alpha=1.0, tol=1e-5, max_steps=1000):

    distort_set = (([], [], []),
                    ([], [], []))

    to_numpy = lambda t: t.detach().cpu().squeeze(0)\
                            .permute(1, 2, 0).numpy()

    for image in input_set:
        image = image.unsqueeze(0)
        eig_obj = Eigendistortion(base_signal=image,
                                  model=model)

        eigdist = eig_obj.synthesize(method='power',
                                     tol=tol,
                                     max_steps=max_steps)[0]

        for idx in range(2):
            eig_dir = eigdist[idx, :].unsqueeze(0)
            distort_set[idx][0].append(to_numpy(eig_dir))

            dist = image + alpha * eig_dir
            distort_set[idx][1].append(to_numpy(dist))

            output = model(dist)
            distort_set[idx][2].append(to_numpy(output))

    return distort_set

# pair difference maximization
def max_pair(model, render, stimuli, n_iter, opt_norm=0.1, stride=0,
            h_init=0.25, beta=0.25, sig_end=0.01, t_max=35, distance=MSE):

    seed = np.random.randint(0, 2**32)
    recon = lambda input: linear_inverse(model, render, input, h_init=h_init,
                                    beta=beta, sig_end=sig_end, stride=0,
                                    seed=seed, t_max=t_max, with_grad=True)[0]

    for n in range(int(n_iter)):
        # clear gradient
        stimuli.clear_grad()
        stim_1, stim_2 = stimuli.get_stimulus()

        # compute reconstruction
        recon_1 = recon(stim_1)
        recon_2 = recon(stim_2)

        loss = distance(recon_1, recon_2)
        loss.backward()

        grad_norm = stimuli.optim_step(opt_norm)

        if stride != 0 and n % stride == 0:
            print('iter: %d, norm: %.4f, loss: %.4f' %
                    (n, grad_norm, loss.item()))

    return stimuli, recon_1, recon_2

# helper object for fill-in optimization
# base class
class FillIn():
    def __init__(self, im_size, device, radius=0.25, init=None, stimuli=False):
        self.im_size = im_size
        self.device = device

        if stimuli:
            if init is None:
                init = np.random.rand(3, *im_size)
            self.stim = torch.tensor(init.astype(np.float32),
                                     requires_grad=True,
                                     device=self.device)

            self.mask, self.flip = self._mask(radius=radius)

    def _mask(self, radius=0.5):
        pass

    def _generator_fn(self, mask, flip):
        return lambda t: t * mask + flip * 0.5

    def _distance_fn(self, flip, spatial=False):
        if spatial:
            return lambda x, y: MSE((x * flip).mean(dim=0),
                                    (y * flip).mean(dim=0))
        else:
            return lambda x, y: MSE(x * flip, y * flip)

    def get_objective(self, radius=0.5, spatial=False):
        mask, flip = self._mask(radius)
        generator = self._generator_fn(mask, flip)
        distance = self._distance_fn(flip, spatial)

        return generator, distance

    def get_stimulus(self):
        stim_1 = self.stim[1] * self.mask + self.stim[0] * self.flip
        stim_2 = self.stim[2] * self.mask + self.stim[0] * self.flip

        return (stim_1, stim_2)

    def clear_grad(self):
        self.stim.grad = None

    def optim_step(self, opt_norm):
        with torch.no_grad():
            grad_norm = torch.norm(self.stim.grad)
            self.stim += self.stim.grad * opt_norm / grad_norm
            self.stim.clamp_(0.0, 1.0)

        return grad_norm

    def show_recon(self, inverse):
        stim = self.get_stimulus()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        for idx, ax in enumerate(axs.flatten()):
            ax.imshow(gamma_correct(stim[idx].detach()\
                      .cpu().permute(1, 2, 0).numpy()))
            ax.axis('off')
        fig.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        for idx, ax in enumerate(axs.flatten()):
            recon = inverse(stim[idx])
            ax.imshow(gamma_correct(recon))
            ax.axis('off')
        fig.show()

class FillInCircle(FillIn):
    def _mask(self, radius=0.5):
        mask = np.ones(shape=self.im_size)

        x, y = np.meshgrid(np.linspace(-1, 1, num=mask.shape[1]),
                           np.linspace(1, -1, num=mask.shape[2]))
        indice = (np.sqrt(x ** 2 + y ** 2)) < radius

        mask[:, indice] = 0.0
        mask = torch.from_numpy(mask.astype(np.float32))\
                                .to(self.device)
        flip = (1.0 - mask).to(self.device)
        return mask, flip

class FillInSquare(FillIn):
    def _mask(self, radius=0.5):
        mask = np.ones(shape=self.im_size)
        edge_len = self.im_size[1]

        idx_lb = int(edge_len * (1 - radius) / 2)
        idx_ub = int(edge_len * (1 + radius) / 2)

        mask[:, idx_lb:idx_ub, idx_lb:idx_ub] = 0.0
        mask = torch.from_numpy(mask.astype(np.float32))\
                                .to(self.device)
        flip = (1.0 - mask).to(self.device)
        return mask, flip