from hmac import new
from re import S
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2_ops import pointnet2_utils

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def sort_sample(points, stride):
    """
    Sort points by distance and sample using stride.

    Args:
        points (torch.Tensor): Tensor of shape [B, D, N], where
            B is the batch size,
            D is the dimensionality of points,
            N is the number of points.
        stride (int): Stride for sampling.

    Returns:
        torch.Tensor: Sorted and sampled points of shape [B, D, M],
                      where M = N // stride.
    """
    B, N, C = points.shape
    
    # Compute distance along the feature dimension (D)
    distance = torch.norm(points, dim=-2)  # Shape: [B, N]
    
    # Sort distances and get the sorting indices
    sorted_indices = torch.argsort(distance, dim=-1)  # Shape: [B, N]
    idx = sorted_indices[:, ::stride]  # Shape: [B, M]
    
    return idx

class PointNorm(nn.Module):
    def __init__(self,in_channel):
        super(PointNorm,self).__init__()
        self.in_channel = in_channel
        self.gamma = nn.Parameter(torch.ones(in_channel))
        self.beta = nn.Parameter(torch.zeros(in_channel))
        self.eps = 1e-6
    def forward(self,x):
        B,N,K,D = x.shape
        mean = x[:,:,0,:].unsqueeze(-2) #[b,n,1,d]
        std = torch.std((x-mean).reshape(B,N,K*D),dim=-1,unbiased=False) #[b,n]
        std = std.unsqueeze(-1).unsqueeze(-1) #[b,n,1,1]
        x = (x-mean)/(std+self.eps)
        x = self.gamma * x + self.beta
        return x

class PointResBlock(nn.Module):
    def __init__(self, in_channel, block_num=2, knn=1, dilation=1):
        super(PointResBlock, self).__init__()
        self.knn = knn
        self.in_channel = in_channel
        self.block_num = block_num
        self.norm = nn.ModuleList([PointNorm(in_channel) for _ in range(block_num)])
        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.MaxPool2d((1,knn)),
        ) for _ in range(block_num)])
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        points, xyz = x
        B, N, C = xyz.shape
        # Compute KNN indices once
        idx = knn_point(self.knn, xyz, xyz)

        for i in range(self.block_num):
            points = points.permute(0, 2, 1)
            grouped_points = index_points(points, idx)  # Group points based on KNN
            grouped_points = self.norm[i](grouped_points) + points.unsqueeze(-2)
            grouped_points = grouped_points.permute(0, 3, 1, 2)
            new_points = self.conv[i](grouped_points).squeeze(-1)
            points = self.relu(new_points + points.permute(0, 2, 1))  # Residual connection with activation

        return points, xyz
    
class PointConv(nn.Module):
    def __init__(self,in_channel,out_channel,knn=1,stride=1,dilation=1):
        super(PointConv,self).__init__()
        self.knn = knn
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm = PointNorm(in_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d((1,knn)),
        )
        if in_channel == out_channel:
            self.identity = nn.Sequential()
        else:
            self.identity = nn.Sequential(
                nn.Conv1d(in_channel,out_channel,kernel_size=1),
                nn.BatchNorm1d(out_channel),
          )
        self.relu = nn.ReLU(inplace=True)
        
        self.blocks = PointResBlock(out_channel,block_num=2,knn=3,dilation=dilation)
        
    def forward(self,x):
        points,xyz = x
        B, N, C = xyz.shape 
        xyz = xyz.contiguous() 
        # fps_idx = sort_sample(points,self.stride)
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, N//self.stride).long()
        sampled_xyz = index_points(xyz, fps_idx)
        sampled_points = index_points(points.permute(0,2,1), fps_idx)

        idx = knn_point(self.knn, xyz, sampled_xyz)
        grouped_points = index_points(points.permute(0,2,1), idx)
        grouped_points = self.norm(grouped_points) + sampled_points.unsqueeze(-2)
        grouped_points = grouped_points.permute(0,3,1,2)
        new_points = self.conv(grouped_points).squeeze(-1)
        new_points = self.relu(new_points + self.identity(sampled_points.permute(0,2,1)))
        # return (new_points,sampled_xyz)
        
        return self.blocks((new_points,sampled_xyz))

if __name__ == "__main__":
    pass
    