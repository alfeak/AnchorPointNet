import torch
import torch.nn as nn
from .utils import pointBase_conv,gen_curve

class PointAnchorNet(nn.Module):
    def __init__(self):
        super(PointAnchorNet, self).__init__()
        self.knn = 24

        self.layer1 = pointBase_conv(3, 64,self.knn,3)
        self.layer2 = pointBase_conv(64,128,6,2,False)
        self.layer3 = pointBase_conv(128,256,6,1,False)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, 40)
        self.batch_list = dict()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if B not in self.batch_list.keys():
            self.batch_list[B] = [gen_curve(B,1,xyz.device),\
                                  gen_curve(B,2,xyz.device),\
                                  gen_curve(B,3,xyz.device)]

        xyz = self.layer1(xyz, self.batch_list[B])
        # print("xyz: ",xyz.shape)
        xyz = self.layer2(xyz, self.batch_list[B])
        xyz = self.layer3(xyz, self.batch_list[B])
        x = self.pool(xyz).squeeze()
        x = self.fc(x)

        return x
if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = PointAnchorNet()
    out = model(data)
    print(out.shape)

