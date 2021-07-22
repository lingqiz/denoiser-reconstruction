import torch, numpy as np
from torch.nn import MSELoss
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

        # compute reconstruction and error
        loss = torch.zeros(1)
        for mtx in render:
            recon, t, _ = linear_inverse(model, mtx, image_in,
                            h_init=h_init, beta=beta, sig_end=sig_end,
                            stride=0, seed=seed, t_max=t_max, with_grad=True)

            # compute the distance between reconstruction and input
            loss += distance(recon, image_in)

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

    return init, recon, sequence

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

# helper object for fill-in optimization
class FillIn():
    def __init__(self, im_size, device):
        self.im_size = im_size
        self.device = device

    def _mask(self, radius=0.5):
        mask = np.ones(shape=self.im_size)

        x, y = np.meshgrid(np.linspace(-1, 1, num=mask.shape[1]),
                           np.linspace(1, -1, num=mask.shape[2]))
        indice = (np.sqrt(x ** 2 + y ** 2)) < radius

        mask[:, indice] = 0.0
        mask = torch.from_numpy(mask.astype(np.float32))\
                                .to(self.device)
        flip = (1.0 - mask).to(self.device)
        return (mask, flip)

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
