import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20):
    idx = knn(x, k=k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims)  
                                       # -> (batch_size*num_points, num_dims) 
                                       #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature

class DGCNN(nn.Module):

    def __init__(
        self, 
        emb_dims=512,
        use_bn=False,
        output_channels=100 # number of categories to predict 
    ):

        super().__init__()

        if use_bn:
            print("using batch norm")
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm2d(emb_dims)

            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv2d(512, emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))
            self.conv5 = nn.Sequential(nn.Conv2d(512, emb_dims, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)


    def forward(self, x):
        batch_size, num_dims, num_points = x.size()                 # x:      batch x   3 x num of points
        x = get_graph_feature(x)                                    # x:      batch x   6 x num of points x 20

        x1     = self.conv1(x)                                      # x1:     batch x  64 x num of points x 20
        x1_max = x1.max(dim=-1, keepdim=True)[0]                    # x1_max: batch x  64 x num of points x 1

        x2     = self.conv2(x1)                                     # x2:     batch x  64 x num of points x 20
        x2_max = x2.max(dim=-1, keepdim=True)[0]                    # x2_max: batch x  64 x num of points x 1

        x3     = self.conv3(x2)                                     # x3:     batch x 128 x num of points x 20
        x3_max = x3.max(dim=-1, keepdim=True)[0]                    # x3_max: batch x 128 x num of points x 1

        x4     = self.conv4(x3)                                     # x4:     batch x 256 x num of points x 20
        x4_max = x4.max(dim=-1, keepdim=True)[0]                    # x4_max: batch x 256 x num of points x 1
 
        x_max  = torch.cat((x1_max, x2_max, x3_max, x4_max), dim=1) # x_max:  batch x 512 x num of points x 1

        point_feat = torch.squeeze(self.conv5(x_max), dim=3)        # point feat:  batch x 512 x num of points

        #global_feat = point_feat.max(dim=2, keepdim=False)[0]       # global feat: batch x 512

        x1 = F.adaptive_max_pool1d(point_feat, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(point_feat, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)


        return x

    def get_global_feature(self, x):
        batch_size, num_dims, num_points = x.size()                 # x:      batch x   3 x num of points
        x = get_graph_feature(x)                                    # x:      batch x   6 x num of points x 20

        x1     = self.conv1(x)                                      # x1:     batch x  64 x num of points x 20
        x1_max = x1.max(dim=-1, keepdim=True)[0]                    # x1_max: batch x  64 x num of points x 1

        x2     = self.conv2(x1)                                     # x2:     batch x  64 x num of points x 20
        x2_max = x2.max(dim=-1, keepdim=True)[0]                    # x2_max: batch x  64 x num of points x 1

        x3     = self.conv3(x2)                                     # x3:     batch x 128 x num of points x 20
        x3_max = x3.max(dim=-1, keepdim=True)[0]                    # x3_max: batch x 128 x num of points x 1

        x4     = self.conv4(x3)                                     # x4:     batch x 256 x num of points x 20
        x4_max = x4.max(dim=-1, keepdim=True)[0]                    # x4_max: batch x 256 x num of points x 1
 
        x_max  = torch.cat((x1_max, x2_max, x3_max, x4_max), dim=1) # x_max:  batch x 512 x num of points x 1

        point_feat = torch.squeeze(self.conv5(x_max), dim=3)        # point feat:  batch x 512 x num of points

        #global_feat = point_feat.max(dim=2, keepdim=False)[0]       # global feat: batch x 512

        x1 = F.adaptive_max_pool1d(point_feat, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(point_feat, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        return x