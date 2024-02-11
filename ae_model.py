import torch
import torch.nn as nn
from torch.nn.functional import normalize
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.lin1 = nn.Linear(54,16)
        self.lin2 = nn.Linear(32,16)
        self.lin3 = nn.Linear(16, 8)
        self.ilin1 = nn.Linear(32, 40)
        self.ilin2 = nn.Linear(16, 54)
        self.ilin3 = nn.Linear(8,16)

        # self.lin4 = nn.Linear(6, 4)
        # self.ilin1 = nn.Linear(10,13)
        # self.ilin2 = nn.Linear(8,10)
        # self.ilin3 = nn.Linear(6, 8)
        # self.ilin4 = nn.Linear(4, 6)
        self.rel =nn.ReLU()
        # self.b0 =nn.BatchNorm1d( 10)
        self.bn00 = nn.BatchNorm1d(40)
        self.bn11 = nn.BatchNorm1d(32)
        self.bn22 = nn.BatchNorm1d(16)
        self.bn33 = nn.BatchNorm1d(8)
        self.bn44 = nn.BatchNorm1d(4)
    def enc(self,x0)   :

        x0=x0.squeeze()
        x1 =x0
        x2= torch.unsqueeze(torch.norm(x1,dim=1),dim=1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = normalize( x1, p=2.0, dim=1)

        # x1 =torch.cat((x1,x2),dim=1)
        x= self.lin1(x1)
        # x = self.b0(x)
        # x=self.rel(self.bn11(x ))
        x = self.rel( x)
        x = self.lin3(x)
        # x = nn.BatchNorm1d(6)
        # x=self.rel(self.bn33(x ))
        x = self.rel(x)

        return x, x1
        x= self.lin2(x)

        # x = nn.BatchNorm1d(8)
        # x=self.rel(self.bn22(x ))
        x=self.rel( x  )

        x = self.lin3(x)
        # x = nn.BatchNorm1d(6)
        # x=self.rel(self.bn33(x ))
        x = self.rel(x)

        # x = self.lin4(x)
        # # x = nn.BatchNorm1d(4)
        # x=self.rel(self.bn44(x ))
        return x,x1
    def dec (self,x):
        # x = self.ilin4(x)
        # X = self.rel(self.bn33(x))
        x = self.ilin3(x)
        x = self.ilin2(x)
        return x
        x = self.rel(self.bn22(x))
        x = self.ilin2(x)
        x = self.rel(self.bn11(x))
        x =self.ilin1(x)
        x = self.rel(self.bn00(x))
        return x
    def forward(self, x):
        encode,x1  = self.enc(x)
        decode = self.dec(encode)
        return decode,x1