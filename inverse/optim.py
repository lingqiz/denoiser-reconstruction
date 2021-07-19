import torch
from torch.nn import MSELoss
from solver import linear_inverse

MSE = MSELoss(reduction='mean')
IDENTITY = lambda x: x
CLAMP= lambda x: x.clamp_(0, 1)

H_INIT = 0.50
BETA = 0.25

def max_diff(model, render, init, n_iter, opt_norm=0.01, stride=0,
            distance=MSE, generator=IDENTITY, constraint=CLAMP):

    sequence = []
    for n in range(int(n_iter)):
        # clear gradient
        init.grad = None

        image_in = generator(init)
        recon, t, _ = linear_inverse(model, render, image_in,
                        h_init=H_INIT, beta=BETA, stride=0, with_grad=True)

        # compute the distance between reconstruction and input
        loss = distance(recon, image_in)
        loss.backward()

        grad_norm = torch.norm(init.grad)

        if stride != 0 and n % stride == 0:
            print('iter: %d, n_step: %d, norm: %.4f, loss: %.4f' % (n, t, grad_norm, loss.item()))
            sequence.append(init.detach().cpu())

        # increase the distance
        with torch.no_grad():
            init += opt_norm / grad_norm * init.grad
            constraint(init)

    return init, recon, sequence