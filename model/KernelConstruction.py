import torch
import torch.nn as nn
class KernelConstruction(nn.Module):
    def __init__(self, kernel_size = 3, outC = 3):
        super(KernelConstruction, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(3,3),padding=1)
        self.outC = outC
        self.norm = nn.Softmax(dim=1)
    def forward(self, x):
        # x (B,IM layers, outH, outW)
        batch_size = x.size(0)
        im_layer = x.size(1)
        outH = x.size(2)
        outW = x.size(3)
        out = self.unfold(x)
        out = out.view(batch_size, im_layer*self.kernel_size*self.kernel_size,outH,outW).contiguous()
        # print("out", out.shape)
        layer_per_out = out.shape[1] // self.outC
        # print("layer_per_out", layer_per_out)
        out_kernels = []
        for i in range(self.outC):
            out_kernels.append(self.norm(out[:,i*layer_per_out:(i+1)*layer_per_out,:,:]))
        # for i in range(len(out_kernels)):
        #     print("kerneL:", out_kernels[i].shape)
        out = torch.cat([out_kernels[i] for i in range(len(out_kernels))],dim=1)
        return out

if __name__ == "__main__":
    input_feature = 18
    N = 8
    H = 224
    W = 224

    model = KernelConstruction(3,3)
    data = torch.randn((N,input_feature,H,W))

    model.cuda()
    data = data.cuda()
    print("data", data.shape)
    feat = model(data)
    print("out", feat.shape)