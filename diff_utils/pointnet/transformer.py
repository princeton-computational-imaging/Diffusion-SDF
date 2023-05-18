import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad



##-----------------------------------------------------------------------------
# Class for Transformer. Subclasses PyTorch's own "nn" module
#
# Computes a KxK affine transform from the input data to transform inputs
# to a "canonical view"
##
class Transformer(nn.Module):

	def __init__(self, num_points=2000, K=3):
		# Call the super constructor
		super(Transformer, self).__init__()

		# Number of dimensions of the data
		self.K = K

		# Size of input
		self.N = num_points

		# Initialize identity matrix on the GPU (do this here so it only 
		# happens once)
		self.identity = grad.Variable(
			torch.eye(self.K).double().view(-1).cuda())

		# First embedding block
		self.block1 =nn.Sequential(
			nn.Conv1d(K, 64, 1),
			nn.BatchNorm1d(64),
			nn.ReLU())

		# Second embedding block
		self.block2 =nn.Sequential(
			nn.Conv1d(64, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU())

		# Third embedding block
		self.block3 =nn.Sequential(
			nn.Conv1d(128, 1024, 1),
			nn.BatchNorm1d(1024),
			nn.ReLU())

		# Multilayer perceptron
		self.mlp = nn.Sequential(
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Linear(256, K * K))


	# Take as input a B x K x N matrix of B batches of N points with K 
	# dimensions
	def forward(self, x):

		# Compute the feature extractions
		# Output should ultimately be B x 1024 x N
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)

		# Pool over the number of points
		# Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
		x = F.max_pool1d(x, self.N).squeeze(2)
		
		# Run the pooled features through the multi-layer perceptron
		# Output should be B x K^2
		x = self.mlp(x)

		# Add identity matrix to transform
		# Output is still B x K^2 (broadcasting takes care of batch dimension)
		x += self.identity

		# Reshape the output into B x K x K affine transformation matrices
		x = x.view(-1, self.K, self.K)

		return x

