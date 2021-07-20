import torch
from torch.nn import MSELoss
from inverse.solver import linear_inverse
from plenoptic.synthesize.eigendistortion import Eigendistortion

# constants associated with the function
MSE = MSELoss(reduction='mean')
IDENTITY = lambda x: x
CLAMP= lambda x: x.clamp_(0, 1)
H_INIT = 0.50
BETA = 0.25

# difference maximization
def max_diff(model, render, init, n_iter, opt_norm=0.01, stride=0,
            distance=MSE, generator=IDENTITY, constraint=CLAMP):

    sequence = []
    for n in range(int(n_iter)):
        # clear gradient
        init.grad = None

        image_in = generator(init)
        recon, t, _ = linear_inverse(model, render, image_in,
                        h_init=H_INIT, beta=BETA, stride=0, seed=0, with_grad=True)

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

# eigendistortion
def eig_distort(model, input_set, alpha=1.0, max_steps=1000):

    distort_set = (([], [], []),
                    ([], [], []))

    for image in input_set:
        image = image.unsqueeze(0)
        eig_obj = Eigendistortion(base_signal=image,
                                  model=model)

        eigdist = eig_obj.synthesize(method='power',
                                     tol=1e-5,
                                     max_steps=max_steps)[0]

    to_numpy = lambda t: t.detach().cpu().squeeze(0)\
                            .permute(1, 2, 0).numpy()

    for idx in range(2):
        eig_dir = eigdist[idx, :].unsqueeze(0)
        distort_set[idx][0].append(to_numpy(eig_dir))

        dist = image + alpha * eig_dir
        distort_set[idx][1].append(to_numpy(dist))

        output = model(dist)
        distort_set[idx][2].append(to_numpy(output))

    return distort_set
