#!/usr/bin/env python
from __future__ import print_function
from itertools import count
import os
import numpy as np
import argparse
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Pytorch Linear Regression')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outf', default='/output',
                    help='folder to output images and model checkpoints')
parser.add_argument('--ckpf', default='',
                    help="path to model checkpoint file (to continue training)")
parser.add_argument('--degree', type=int, default=4, metavar='P',
                    help='polynomial degree to learn(default: 4)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--train', action='store_true',
                    help='training a fully connected layer')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate a [pre]trained model from a random tensor.')

args = parser.parse_args()

# Is there the outf?
try:
    os.makedirs(args.outf)
except OSError:
    pass

# Is CUDA available?
cuda = torch.cuda.is_available()
# Seed for replicability
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)


POLY_DEGREE = args.degree
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    # Build samples from a normal distribution with zero mean
    # and variance of one.
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


# Define model
fc = torch.nn.Linear(W_target.size(0), 1)
if cuda:
    fc.cuda()

# Load checkpoint
if args.ckpf != '':
    if cuda:
        fc.load_state_dict(torch.load(args.ckpf))
    else:
        # Load GPU model on CPU
        fc.load_state_dict(torch.load(args.ckpf, map_location=lambda storage, loc: storage))
        fc.cpu()

# Check if model use cuda
#print (next(fc.parameters()).is_cuda)

# Train?
if args.train:
    # Iterate until the loss is under 1e-3 threshold
    for batch_idx in count(1):
        fc.train()
        # Get data
        batch_x, batch_y = get_batch(args.batch_size)

        if cuda:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        # Reset gradients
        fc.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(fc(batch_x), batch_y)
        loss = output.data[0]

        # Backward pass
        output.backward()

        # Apply gradients (SGD with learning_rate=0.1 and batch_size=32)
        for param in fc.parameters():
            param.data.add_(-0.1 * param.grad.data)

        # Stop criterion
        if loss < 1e-3:
            break

    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
    # Do checkpointing - Is saved in outf
    torch.save(fc.state_dict(), '%s/regression_%d_degree_polynomial.pth' % (args.outf, args.degree))

# Evaluate?
if args.evaluate:
    fc.eval()
    # Custom Tensor
    # t_test = torch.Tensor([[... ** i for i in range(1, POLY_DEGREE+1)]])
    # v_test = Variable(t_test)
    # print (v_test.size())
    # print('==> Actual function result:\t' + str(f(t_test.cpu())))
    x_test, y_test = get_batch(batch_size=1)
    if cuda:
        x_test = x_test.cuda()
    # Comparison
    print ('==> Learned function result:\t' + str(fc(x_test)))
    print('==> Actual function result:\t' + str(y_test))
