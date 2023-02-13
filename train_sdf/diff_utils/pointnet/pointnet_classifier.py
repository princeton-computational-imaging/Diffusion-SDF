import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as grad

from .pointnet_base import PointNetBase


##-----------------------------------------------------------------------------
# Class for PointNetClassifier. Subclasses PyTorch's own "nn" module
#
# Computes the local embeddings and global features for an input set of points
##
class PointNetClassifier(nn.Module):

	def __init__(self, num_points=2000, K=3):
		# Call the super constructor
		super(PointNetClassifier, self).__init__()

		# Local and global feature extractor for PointNet
		self.base = PointNetBase(num_points, K)

		# Classifier for ShapeNet
		self.classifier = nn.Sequential(
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Dropout(0.7),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(0.7),
			nn.Linear(256, 40))


	# Take as input a B x K x N matrix of B batches of N points with K 
	# dimensions
	def forward(self, x):

		# Only need to keep the global feature descriptors for classification
		# Output should be B x 1024
		global_feature, local_embedding, T2 = self.base(x)

		# first attempt: only use the global feature
		#return global_feature

		# second attempt: concat local and global feature similar to segmentation network but create the downsampling MLP in the diffusion model
		# local embedding shape: B x 64 x N; global feature shape: B x 1024; concat to get B x 1088 x N 
		num_points = local_embedding.shape[-1]
		global_feature = global_feature.unsqueeze(-1).repeat(1,1,num_points)
		point_features = torch.cat( (global_feature, local_embedding), dim=1 ) # shape is B x 1088 x N 
		return point_features
		
		

		# Returns a B x 40 
		#return self.classifier(x), T2