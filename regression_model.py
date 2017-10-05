from __future__ import print_function
from itertools import count
import os
import numpy as np
import argparse
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable


class LinearRegression(object):
	"""Liner Regression Model"""
	def __init__(self, ckp="/input/regression_4_degree_polynomial.pth", degree=4, batch_size=1):
		# Degree To fit
		self._degree = degree
		self._batch_size = batch_size
		# Use CUDA?
		self._cuda = torch.cuda.is_available()
		try:
			os.path.isfile(ckp)
			self._ckp = ckp
		except IOError as e:
			# Does not exist OR no read permissions
			print ("Unable to open ckp file")

	def _make_features(self, x):
		"""Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
		x = x.unsqueeze(1)
		return torch.cat([x ** i for i in range(1, self._degree+1)], 1)

	def _poly_desc(self, W, b):
		"""Creates a string description of a polynomial."""
		result = 'y = '
		for i, w in enumerate(W):
			result += '{:+.2f} x^{} '.format(w, len(W) - i)
		result += '{:+.2f}'.format(b[0])
		return result

	def _get_batch(self, batch_size=1):
		"""Builds a batch i.e. (x, f(x)) pair."""
		# Build samples from a normal distribution with zero mean
		# and variance of one.
		random = torch.randn(batch_size)
		x = self._make_features(random)
		return Variable(x)

	def build_model(self):
		# Define model
		self._model = torch.nn.Linear(self._degree, 1)
		if self._cuda:
			self._model.cuda()
		# Load checkpoint
		if self._ckp != '':
			if self._cuda:
				self._model.load_state_dict(torch.load(self._ckp))
			else:
				# Load GPU model on CPU
				self._model.load_state_dict(torch.load(self._ckp, map_location=lambda storage, loc: storage))
				self._model.cpu()

	def evaluate(self):
		self._model.eval()
		x_test = self._get_batch(batch_size=self._batch_size)
		if self._cuda:
			x_test = x_test.cuda()
		learned = self._poly_desc(self._model.weight.data.view(-1),
				self._model.bias.data)
		output = np.asscalar(self._model(x_test).data.cpu().numpy())
		return '==> Learned function result: {l}\n==> Data: {d}\n==> Output: {o}'.format(l=learned, d=x_test.data, o=output)
