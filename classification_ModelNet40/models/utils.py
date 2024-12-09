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
    distance = torch.norm(points, dim=-1)  # Shape: [B, N]
    
    # Sort distances and get the sorting indices
    sorted_indices = torch.argsort(distance, dim=-1)  # Shape: [B, N]
    idx = sorted_indices[:, ::stride]  # Shape: [B, M]
    
    return idx
   
class PointMaxPool(nn.Module):
    def __init__(self,channel,knn=1,stride=1,dilation=1):
        super(PointMaxPool,self).__init__()
        self.knn = knn
        self.stride = stride
        self.dilation = dilation
        self.channel = channel
        self.pool = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.MaxPool2d((1,knn)),
            nn.BatchNorm2d(channel),
        )

    def forward(self,x):
        points,xyz = x
        B, N, C = xyz.shape
        xyz = xyz.contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, N//self.stride).long()
        sampled_xyz = index_points(xyz, fps_idx)
        sampled_points = index_points(points.permute(0,2,1), fps_idx).permute(0,2,1)

        idx = knn_point(self.knn, xyz, sampled_xyz)
        grouped_points = index_points(points.permute(0,2,1), idx)
        grouped_points = grouped_points.permute(0,3,1,2)
        new_points = self.pool(grouped_points).squeeze(-1)
        
        return new_points, sampled_xyz
    
class PointConv(nn.Module):
    def __init__(self,in_channel,out_channel,knn=1,stride=1,dilation=1):
        super(PointConv,self).__init__()
        self.knn = knn
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False),
            nn.MaxPool2d((1,knn)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel,out_channel,kernel_size=1,bias=False),
            nn.MaxPool2d((1,knn)),
            nn.BatchNorm2d(out_channel),
        )

        self.identity = nn.Sequential(
                nn.Conv1d(in_channel,out_channel,kernel_size=1,bias=False),
                nn.BatchNorm1d(out_channel),
            )
        
    def forward(self,x):
        points,xyz = x
        B, N, C = xyz.shape
        xyz = xyz.contiguous() 
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, N//self.stride).long()
        sampled_xyz = index_points(xyz, fps_idx)
        sampled_points = index_points(points.permute(0,2,1), fps_idx).permute(0,2,1)

        idx = knn_point(self.knn, xyz, sampled_xyz)
        grouped_points = index_points(points.permute(0,2,1), idx)
        grouped_points = grouped_points.permute(0,3,1,2)
        new_points = self.conv(grouped_points).squeeze(-1)

        idx = knn_point(self.knn, sampled_xyz, sampled_xyz)
        grouped_points = index_points(new_points.permute(0,2,1), idx)
        grouped_points = grouped_points.permute(0,3,1,2)
        new_points = self.conv1(grouped_points).squeeze(-1)

        new_points = F.relu(new_points + self.identity(sampled_points))
        return (new_points,sampled_xyz)
        
if __name__ == "__main__":
    pass
    