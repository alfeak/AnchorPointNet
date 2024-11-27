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
    B, D, N = points.shape
    
    # Compute distance along the feature dimension (D)
    distance = torch.norm(points, dim=1)  # Shape: [B, N]
    
    # Sort distances and get the sorting indices
    sorted_indices = torch.argsort(distance, dim=-1)  # Shape: [B, N]
    idx = sorted_indices[:, ::stride]  # Shape: [B, M]
    
    return idx

class PointMaxPool(nn.Module):
    def __init__(self,in_channel,knn=1,stride=1,dilation=1):
        super(PointMaxPool,self).__init__()
        self.knn = knn
        self.stride = stride
        self.dilation = dilation
        self.pool = nn.MaxPool2d((1,self.knn))
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self,x):
        points,xyz = x
        # fps_idx = sort_sample(points,self.stride)
        points = points.permute(0,2,1)
        B, N, C = xyz.shape
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, N//self.stride).long()
        
        sampled_xyz = index_points(xyz, fps_idx)
        sampled_points = index_points(points, fps_idx)

        idx = knn_point(self.knn * self.dilation, xyz, sampled_xyz)[:, :, ::self.dilation]
        new_points = self.pool(self.bn(index_points(points, idx).permute(0,3,1,2))).squeeze(-1)
        return (new_points, sampled_xyz)

class PointBatchNorm(nn.Module):
    def __init__(self,in_channel,eps=1e-5):
        super(PointBatchNorm,self).__init__()
        self.in_channel = in_channel
        self.affine_alpha = nn.Parameter(torch.ones([1,in_channel,1,1]))
        self.affine_beta = nn.Parameter(torch.zeros([1,in_channel,1,1]))
        self.eps = eps
    
    def forward(self,points):
        B,D,N,K = points.shape
        anchor_points = points[:,:,:,0].unsqueeze(-1).repeat(1,1,1,K)
        var = torch.var((points-anchor_points).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        std = torch.sqrt(var+self.eps)
        points = (points-anchor_points)/std
        points = self.affine_alpha*points + self.affine_beta
        
        return points

class PointFullAgreggation(nn.Module):
    def __init__(self):
        super(PointFullAgreggation,self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self,x):
        points,xyz = x
        points = points.permute(0,2,1)
        B, N, C = xyz.shape
        idx = knn_point(N, xyz, xyz)
        grouped_points = index_points(points, idx).permute(0,3,1,2)  # [B, C, N, nsample]
        new_points = F.max_pool2d(grouped_points, kernel_size=(1,N)).squeeze(-1)  # [B, C, N]
        new_points = self.pool(new_points).squeeze(-1)  # [B, C]
        return (new_points, xyz)

class PointConv(nn.Module):
    def __init__(self,in_channel,out_channel,knn=1,stride=1,dilation=1,activate=True):
        super(PointConv,self).__init__()
        self.knn = knn
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.activate = activate
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d((1,self.knn)),
        )

    def forward(self,x):
        points,xyz = x
        B, N, C = xyz.shape
        xyz = xyz.contiguous() 
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, N//self.stride).long()
        # fps_idx = sort_sample(points,self.stride)

        sampled_xyz = index_points(xyz, fps_idx)
        sampled_points = index_points(points.permute(0,2,1),fps_idx).permute(0,2,1)
        
        idx = knn_point(self.knn * self.dilation, xyz, sampled_xyz)[:, :, ::self.dilation]
        grouped_points = index_points(points.permute(0,2,1),idx).permute(0,3,1,2)
        new_points = self.bn(grouped_points) + sampled_points.unsqueeze(-1).repeat(1,1,1,self.knn)
        new_points = self.conv(new_points).squeeze(-1)
        if self.activate:
            new_points = F.relu(new_points)
        return (new_points,sampled_xyz)

class PointResConv(nn.Module):
    def __init__(self,in_channel,out_channel,knn=1,stride=1,dilation=1):
        super(PointResConv,self).__init__()
        self.knn = knn
        self.stride = stride
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            PointConv(in_channel,out_channel,knn,stride,dilation),
            PointConv(out_channel,out_channel,knn,1,dilation,False)
        )
        self.conv1 = nn.Sequential(
            PointConv(out_channel,out_channel,knn,1,dilation),
            PointConv(out_channel,out_channel,knn,1,dilation,False)
        )
        
        if in_channel == out_channel:
            self.identity = nn.Sequential()
        else:
            self.identity = PointConv(in_channel,out_channel,1,stride,dilation,False)
            
    def forward(self,x):
        points,sampled_xyz = self.conv(x)
        resi,_ = self.identity(x)
        points = F.relu(points + resi)
        
        new_points,_ = self.conv1((points,sampled_xyz))
        new_points = F.relu(new_points + points)
        return (new_points,sampled_xyz)

if __name__ == "__main__":
    pass
    