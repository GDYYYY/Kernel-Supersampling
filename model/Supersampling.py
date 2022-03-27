import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import math
import numpy as np
from PIL import Image, ImageOps

class Supersampling(nn.Module): 
    
    def __init__(self,kernel_size =3, outC = 3, featC = 6):
        super(Supersampling, self).__init__()
        self.kernel_size = kernel_size
        self.outC = outC
        self.featC = featC
        self.unfold = nn.Unfold(kernel_size=(3,3),padding=1)

    def forward(self, feat, kernel_map):
        # kernel_map (N, 6*3*3*3, H, W)
        # inC=6 --> outC=3 kernel_size = 3  
        # kernel对每个像素，有6x3x3个weight值
        # 每个像素有RGB 3个kernel，weight值有6x3x3x3个
        # 总共有 N*H*W*6*3*3*3 (N*outH*outW*inC*outC*k*k)
        ## 变换为
        ## -->
        # kernel_map (N, outH*outW, inC*k*k, outC)

        # feat (N,inC,outH,outW) 
        # feat (N,18,outH,outW) # 6*3
        ## 变换为
        ## -->
        ## feat (N, outH*outW, 1, inC*k*k)
        
        ## matmul (feat, kernel_map) --> (N,outH*outW, 1, outC)
        ## --> (N, outC, outH, outW)
        batch_size = feat.size(0)
        H = feat.size(2)
        W = feat.size(3)

        # print("batch_size ",batch_size)
        # print("H ",H)
        # print("W ",W)

        # N, 6, H, W
        # R = feat[:,0:self.featC,:,:]
        
        # N, 6*9, H, W
        K_R = kernel_map[:,0:self.featC*self.kernel_size*self.kernel_size,:,:] 

        # G = feat[:,self.featC:2*self.layer_per_out,:,:]
        K_G = kernel_map[:,self.featC*self.kernel_size*self.kernel_size:2*self.featC*self.kernel_size*self.kernel_size,:,:] 
        
        # B = feat[:,2*self.featC:3*self.layer_per_out,:,:]
        K_B = kernel_map[:,2*self.featC*self.kernel_size*self.kernel_size:3*self.featC*self.kernel_size*self.kernel_size,:,:] 

        # Demo Code
        # >>> # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
        # >>> inp = torch.randn(1, 3, 10, 12)
        # >>> w = torch.randn(2, 3, 4, 5)
        # >>> inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        # >>> out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        # >>> out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        # >>> # or equivalently (and avoiding a copy),
        # >>> # out = out_unf.view(1, 2, 7, 8)
        # >>> (torch.nn.functional.conv2d(inp, w) - out).abs().max()
        # tensor(1.9073e-06)

        feat_unf = self.unfold(feat)
        feat_mat = feat_unf.permute(0,2,1).reshape(-1, 1, self.featC*self.kernel_size*self.kernel_size)
        # R_unf = self.unfold(R) # N, 6*9, H*W
        # R_mat = N*H*W, 1, 6*9
        # R_mat = R_unf.permute(0,2,1).reshape(-1, 1, self.layer_per_out*self.kernel_size*self.kernel_size)
        # K_R_mat =  N*H*W, 6*9 , 1
        K_R_mat = K_R.permute(0,2,3,1).reshape(-1, self.featC*self.kernel_size*self.kernel_size, 1)
        Out_R = torch.matmul(feat_mat,K_R_mat).squeeze(1).squeeze(1).reshape(batch_size,H,W).unsqueeze(1) # N*H*W

        # G_unf = self.unfold(G) # N, 6*9, H*W
        # R_mat = N*H*W, 1, 6*9
        # G_mat = G_unf.permute(0,2,1).reshape(-1, 1, self.layer_per_out*self.kernel_size*self.kernel_size)
        # K_R_mat =  N*H*W, 6*9 , 1
        K_G_mat = K_G.permute(0,2,3,1).reshape(-1, self.featC*self.kernel_size*self.kernel_size, 1)
        Out_G = torch.matmul(feat_mat,K_G_mat).squeeze(1).squeeze(1).reshape(batch_size,H,W).unsqueeze(1) # N*H*W

        # B_unf = self.unfold(B) # N, 6*9, H*W
        # R_mat = N*H*W, 1, 6*9
        # B_mat = B_unf.permute(0,2,1).reshape(-1, 1, self.layer_per_out*self.kernel_size*self.kernel_size)
        # K_R_mat =  N*H*W, 6*9 , 1
        K_B_mat = K_B.permute(0,2,3,1).reshape(-1, self.featC*self.kernel_size*self.kernel_size, 1)
        Out_B = torch.matmul(feat_mat,K_B_mat).squeeze(1).squeeze(1).reshape(batch_size,H,W).unsqueeze(1) # N*H*W
        
        out = torch.cat((Out_R,Out_G,Out_B),dim=1)
        return out



if __name__ == "__main__":
    input_feature = 6
    N = 8
    H = 224
    W = 224
    kernel_map_channel = 162 # 6x3x3x3
    kernel_size = 3

    model = Supersampling(3,3,6)
    data = torch.randn((N,input_feature,H,W))
    kernel_map = torch.randn((N,kernel_map_channel,H,W))

    model.cuda()
    data = data.cuda()
    kernel_map = kernel_map.cuda()
    print("data", data.shape)
    print("kernel_map", kernel_map.shape)
    feat = model(data,kernel_map)
    print("out", feat.shape)

    ### Image Test
    def load_img(image_path):
        HR = Image.open(image_path+'.png').convert('RGB')
        return HR
    def preprocess1(img):
        img = np.array(img).astype(np.float32)
        img /= 255.
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        return img

    input_feature = 3
    N = 1
    
    kernel_map_channel = 3*3*3*3
    kernel_size =3
    model = Supersampling(3,3,3)

    data = load_img("img_001_SRF")# read image
    data = preprocess1(data)
    print("image:",data.shape)
    data = data.unsqueeze(0)
    H = data.size(2)
    W = data.size(3)

    kernel_map_R = torch.cat((torch.ones((N,9,H,W)) / 9.0, torch.zeros((N,18,H,W))), dim=1)
    print("kernel_map_R:", kernel_map_R.shape)
    kernel_map_G = torch.cat((torch.zeros((N,9,H,W)) , torch.ones((N,9,H,W))/ 9.0, torch.zeros((N,9,H,W))), dim =1)
    print("kernel_map_G:", kernel_map_G.shape)
    kernel_map_B = torch.cat((torch.zeros((N,18,H,W)) , torch.ones((N,9,H,W))/ 9.0), dim =1)
    print("kernel_map_B:", kernel_map_B.shape)
    kernel_map = torch.cat((kernel_map_R,kernel_map_G,kernel_map_B),dim=1)
    print("kernel_map:", kernel_map.shape)
    
    model.cuda()
    data = data.cuda()
    kernel_map = kernel_map.cuda()
    print("data", data.shape)
    print("kernel_map", kernel_map.shape)

    out = model(data,kernel_map)
    print("out", out.shape)

    out = out.cpu()
    # saveImg
    import imageio
    out = out[0,0:3,:,:].numpy() * 255.
    out = out.transpose([1, 2, 0]).astype(np.uint8)
    imageio.imwrite('out.jpg', out)

    ### Image Test End