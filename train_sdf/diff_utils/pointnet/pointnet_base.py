import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from .transformer import Transformer


##-----------------------------------------------------------------------------
# Class for PointNetBase. Subclasses PyTorch's own "nn" module
#
# Computes the local embeddings and global features for an input set of points
##
class PointNetBase(nn.Module):

    def __init__(self, num_points=2000, K=3):
        # Call the super constructor
        super(PointNetBase, self).__init__()

        # Input transformer for K-dimensional input
        # K should be 3 for XYZ coordinates, but can be larger if normals, 
        # colors, etc are included
        self.input_transformer = Transformer(num_points, K)

        # Embedding transformer is always going to be 64 dimensional
        self.embedding_transformer = Transformer(num_points, 64)

		# Multilayer perceptrons with shared weights are implemented as 
		# convolutions. This is because we are mapping from K inputs to 64 
		# outputs, so we can just consider each of the 64 K-dim filters as 
		# describing the weight matrix for each point dimension (X,Y,Z,...) to
		# each index of the 64 dimension embeddings
        self.mlp1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU())


	# Take as input a B x K x N matrix of B batches of N points with K 
	# dimensions
    def forward(self, x):

        # Number of points put into the network
        N = x.shape[2]

        # First compute the input data transform and transform the data
        # T1 is B x K x K and x is B x K x N, so output is B x K x N
        T1 = self.input_transformer(x)
        x = torch.bmm(T1, x)

        # Run the transformed inputs through the first embedding MLP
        # Output is B x 64 x N
        x = self.mlp1(x)

        # Transform the embeddings. This gives us the "local embedding" 
        # referred to in the paper/slides
        # T2 is B x 64 x 64 and x is B x 64 x N, so output is B x 64 x N
        T2 = self.embedding_transformer(x)
        local_embedding = torch.bmm(T2, x)

        # Further embed the "local embeddings"
        # Output is B x 1024 x N
        global_feature = self.mlp2(local_embedding)

        # Pool over the number of points. This results in the "global feature"
        # referred to in the paper/slides
        # Output should be B x 1024 x 1 --> B x 1024 (after squeeze)
        global_feature = F.max_pool1d(global_feature, N).squeeze(2)

        return global_feature, local_embedding, T2

