import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import math
from hilbert import decode

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return group_idx

def gen_curve(batch_num, num_bits,device):
    # Number of dimensions.
    num_dims = 3
    # The maximum Hilbert integer.
    max_h = 2**(num_bits*num_dims)
  # Generate a sequence of Hilbert integers.
    hilberts = np.arange(max_h)
    # Compute the 2-dimensional locations.
    locs = decode(hilberts, num_dims, num_bits)
    locs = locs.astype(np.float32) / (2**num_bits - 1) *2 -1
    anchor_pc = torch.tensor(locs).to(device).unsqueeze(0).repeat(batch_num,1,1)

    return anchor_pc

def gen_anchor_pc1d(batch_num,anchor_pc_num,range,device):
    min_,max_ = range
    anchor_pc = torch.linspace(min_,max_,anchor_pc_num).to(device).unsqueeze(-1).repeat(batch_num,1,1)
    zero_padding = torch.zeros(anchor_pc_num).to(device).unsqueeze(-1).repeat(batch_num,1,1)
    anchor_pc = torch.cat([anchor_pc,zero_padding,zero_padding],dim=-1)
    return anchor_pc

class ChannelAttention1d(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
           
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x0

class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1d, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x0

class pointBase_embedding(nn.Module):
    def __init__(self,in_channel,out_channel, knn):
        super(pointBase_embedding,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel*2,out_channel,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d((1,knn))
        )
        self.affine_alpha = nn.Parameter(torch.ones([1,1,1, in_channel]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channel]))

    def forward(self,xyz):
        B, N ,K, C = xyz.shape
        mean = torch.mean(xyz, dim=2, keepdim=True)
        std = torch.std((xyz-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        new_xyz = (xyz-mean)/(std + 1e-5)
        new_xyz = self.affine_alpha*new_xyz + self.affine_beta
        new_xyz = torch.cat((xyz,new_xyz),dim=-1)
        new_xyz = new_xyz.permute(0,3,1,2)
        new_xyz = self.conv(new_xyz).squeeze()
        return new_xyz

def arrange_pc(xyz,anchor_pc,order,knn,is_first=True):
    # print(order)
    xyz = xyz.permute(0,2,1)
    if is_first:
        # print(xyz.shape,anchor_pc[order-1].shape)
        idx = knn_point(knn,xyz,anchor_pc[order-1])
        # new_xyz = index_points(xyz.permute(0,2,1), idx) #.permute(0,3,1,2)
    else:
        # print(xyz.shape,anchor_pc[order].shape,anchor_pc[order-1].shape)
        idx = knn_point(knn,anchor_pc[order],anchor_pc[order-1])
    new_xyz = index_points(xyz, idx) #.permute(0,3,1,2)
    return new_xyz

class pointBase_conv(nn.Module):
    def __init__(self,in_channel,out_channel,knn=24,order=3,is_first=True,stride=8):
        super(pointBase_conv,self).__init__()
        self.order = order
        self.is_first = is_first
        self.knn = knn
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel*2,out_channel,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d((1,knn)),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(out_channel,out_channel,3,1,1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            ChannelAttention1d(out_channel),
            SpatialAttention1d(),
        )
        self.affine_alpha = nn.Parameter(torch.ones([1,1,1, in_channel]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channel]))
        
    def forward(self,xyz,anchor_pc):
        new_xyz = arrange_pc(xyz,anchor_pc,self.order,self.knn,self.is_first)
        B, N ,K, C = new_xyz.shape
        mean = torch.mean(new_xyz, dim=2, keepdim=True)
        std = torch.std((new_xyz-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        normalized_xyz = (new_xyz-mean)/(std + 1e-5)
        normalized_xyz = self.affine_alpha*normalized_xyz + self.affine_beta
        new_xyz = torch.cat((new_xyz,normalized_xyz),dim=-1)
        new_xyz = new_xyz.permute(0,3,1,2)
        xyz = self.conv(new_xyz).squeeze()
        xyz = xyz + self.conv1(xyz)

        return xyz

if __name__ == "__main__":
    # pass
    xyz = torch.randn(1, 1024,3)
    high_xyz = torch.randn(1, 1024, 6)
    new_xyz = torch.randn(1,24,3)
    new_xyz = arrange_pc(high_xyz,xyz,new_xyz,24)
    print(new_xyz.shape)
    # curve = gen_curve(2,3,xyz.device)
    # print(curve.shape)
    # print(curve[0][:10])