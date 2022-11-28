# Computes the norm of the Jacobian on CIFAR-10 data using code from [1].
# [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
# "Robust Learning with Jacobian Regularization," 2019.
# [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)

from __future__ import division
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision 
import sys
import numpy as np
from pathlib import Path
import json
from models import instantiate_model
from utilities import args_to_model_string, get_args

class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        B,C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v=torch.zeros(B,C)
                v[:,ii]=1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C,B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=False) # added False on 5/6/22 because it's just a stat, not doing reg
            J2 += C*torch.norm(Jv)**2 / (num_proj*B)
        R = (1/2)*J2
        return R

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1: 
            return torch.ones(B)
        v=torch.randn(B,C)
        arxilirary_zero=torch.zeros(B,C)
        vnorm=torch.norm(v, 2, 1,True)
        v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v
                                                                            
    def _jacobian_vector_product(self, y, x, v, create_graph=False): 
        '''
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''                                                            
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v, 
                                        retain_graph=True, 
                                        create_graph=create_graph)
        return grad_x

# the following is an interface to the above code from FAIR
def get_c10_batch(PATH_TO_c10):
    N = 400 # error is order (N*n_proj)^-.5, where n_proj is # of classes if n_proj=-1
    cifar_mean = [0.491, 0.482, 0.447]
    cifar_std = [0.247, 0.243, 0.262]
    testset = torchvision.datasets.CIFAR10(root=PATH_TO_c10, 
                                            train=False,
                                            download=True,
                                            transform=None)
    data = testset.data[:N]  # [200, 32, 32, 3]
    data = (data / 255. - cifar_mean) / cifar_std
    labels = testset.targets[:N]
    return torch.from_numpy(data).permute((0,3,1,2)).float().cuda(), torch.tensor(labels).cuda()

def compute_jacobian_norm(model_string, PATH_TO_RobustNets, PATH_TO_c10):
    # get batch of inputs
    inputs, labels = get_c10_batch(PATH_TO_c10)

    # load model weights
    model = instantiate_model(model_string, PATH_TO_RobustNets).cuda()
    model.eval() # put model into eval mode

    # apply model to dataset and do sanity check on acc
    inputs.requires_grad_()
    # sanity check the accuracy
    outputs = model(inputs)
    half_square_frob_norm_jacobian = JacobianReg(n=-1)(inputs, outputs).item()

    # get jacobian calculator and approximation to its norm
    result = np.sqrt(half_square_frob_norm_jacobian*2) # sqrt times 2 because the J calculator uses 1/2 * J^2

    return result

if __name__=='__main__':
    args = get_args()
    assert args.PATH_TO_c10, 'you must specify a location for the CIFAR-10 data we will create using the arg --PATH_TO_c10'
    args.PATH_TO_c10 = Path(args.PATH_TO_c10)
    model_string = args_to_model_string(args)
    print('Jacobian is: ', compute_jacobian_norm(
        model_string, Path(args.PATH_TO_RobustNets), args.PATH_TO_c10))
    with open('RobustNets/metric_and_OOD_var_dict.json', 'r') as f:
        metric_dict = json.load(f)
    print(f'Metric value in "Models Out of Line..." was {metric_dict[model_string]["jacobian_norm"]}')
