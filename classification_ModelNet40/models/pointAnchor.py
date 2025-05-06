import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils


class PointPreNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels,1,1))
        self.beta = nn.Parameter(torch.zeros(in_channels,1,1))
        self.eps = 1e-6
        
    def forward(self, points):
        anchor = points[:,:,:,0].unsqueeze(-1)
        std = torch.std(points - anchor, dim=1, keepdim=True, unbiased=False)
        points = (points - anchor) / (std + self.eps)

        return points * self.alpha + self.beta

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

class maxpool(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.pool = nn.MaxPool2d((1,k))
    def forward(self,x):
        return self.pool(x).squeeze(-1)

class SetConv(nn.Module):
    def __init__(self, in_channels,out_channels,nsample,prenorm,skip_conn):
        super().__init__()
        self.skip_conn = skip_conn
        self.norm = PointPreNorm(in_channels) if prenorm else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels,out_channels,1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels,out_channels,1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            maxpool(nsample),
        )

        self.act = nn.ReLU(inplace=True)
        self.maxpool = maxpool(nsample)
        
    def forward(self, features):
        features = self.norm(features)
        features = self.conv(features)
        if self.skip_conn:
            features = self.act(self.conv1(features) + self.maxpool(features))
        else:
            features = self.act(self.conv1(self.act(features)))
        return features

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, nsample,prenorm,skip_conn):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.nsample = nsample
        self.conv = nn.Sequential(
            SetConv(in_channels,out_channels,nsample,prenorm,skip_conn),
            nn.Conv1d(out_channels,out_channels,1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels,out_channels,1),
            nn.BatchNorm1d(out_channels),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,1),
            nn.BatchNorm1d(out_channels),
        ) if in_channels!=out_channels else nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, points):
        xyz, features = points
        B, N, C = xyz.shape
        features = features.permute(0,2,1)
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, N//self.stride).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_features = index_points(features, fps_idx).permute(0,2,1)  # [B, npoint, d]

        idx = knn_point(self.nsample, xyz, new_xyz)
        grouped_points = index_points(features, idx).permute(0,3,1,2)  # [B, npoint, k, d]

        new_features = self.act(self.conv(grouped_points) + self.skip_conv(new_features))

        return (new_xyz, new_features)


class PointAnchorNet(nn.Module):
    def __init__(self,num_classes=40,prenorm=True,skip_conn=True):
        super(PointAnchorNet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.Sequential(
            PointConv(64,128,2,24,prenorm,skip_conn),
            PointConv(128,256,2,24,prenorm,skip_conn),
            PointConv(256,512,2,24,prenorm,skip_conn),
            PointConv(512,1024,2,24,prenorm,skip_conn)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N

        xyz,x = self.convs((xyz,x))
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)

        return x

def Net_Orin(num_classes=40, **kwargs) -> PointAnchorNet:
    return PointAnchorNet(num_classes,False,False)

def Net_prenorm(num_classes=40, **kwargs) -> PointAnchorNet:
    return PointAnchorNet(num_classes,True,False)

def Net_maxpoolConn(num_classes=40, **kwargs) -> PointAnchorNet:
    return PointAnchorNet(num_classes,False,True)

def Net(num_classes=40, **kwargs) -> PointAnchorNet:
    return PointAnchorNet(num_classes,True,True)
    
if __name__ == '__main__':
    data = torch.rand(2, 3, 1024).cuda()
    print("===> testing pointMLP ...")
    # model = Net_Orin().cuda()
    # model = Net_prenorm().cuda()
    # model = Net_maxpoolConn().cuda()
    model = Net().cuda()
    print(model)
    out = model(data)
    print(out.shape)

