import torch
import torch.nn as nn
from .utils import PointConv#,PointMaxPool

class PointAnchorNet(nn.Module):
    def __init__(self,knn=24,dilation=1):
        super(PointAnchorNet, self).__init__()
        knn = knn
        dilation = dilation
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.convlayer = nn.Sequential(
            # PointConv(3,64,knn,2,dilation),
            # PointMaxPool(64,knn,2,dilation),
            PointConv(64,128,knn,2,dilation),
            # PointConv(128,128,knn,1,dilation),
            PointConv(128,256,knn,2,dilation),
            # PointConv(256,256,knn,1,dilation),
            PointConv(256,512,knn,2,dilation),
            # PointConv(512,512,knn,1,dilation),
            PointConv(512,1024,knn,2,dilation),
            # PointConv(1024,1024,knn,1,dilation),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(1024, 40)
        
    def forward(self, xyz):
        points = self.conv(xyz.permute(0,2,1))
        points,xyz = self.convlayer((points,xyz))
        # points,xyz = self.convlayer((xyz.permute(0,2,1),xyz))
        points = self.pool(points).squeeze(-1)
        points = self.classifier(points)
        return points
    
if __name__ == '__main__':
    # data = torch.rand(2, 1024, 3).cuda()
    data = torch.rand(2, 1024, 3)
    print("===> testing pointMLP ...")
    # model = PointAnchorNet().cuda()
    model = PointAnchorNet()
    out = model(data)
    print(out.shape)
