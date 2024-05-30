import torch
from torch.nn import functional as F
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}
def LEGM(source_imgs1,source_imgs2)->torch.FloatTensor():
    dimension=source_imgs1.shape
    device=source_imgs1.device
    s1 = source_imgs1 ** 2
    s2 = source_imgs2 ** 2
    weight = torch.ones(size=(dimension[1], 1, 3, 3), device=device, dtype=s1.dtype)
    E_1 = F.conv2d(s1, weight, stride=1, padding=1, groups=dimension[1])
    E_2 = F.conv2d(s2, weight, stride=1, padding=1, groups=dimension[1])
    h_sobel=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=source_imgs1.dtype,device=device).view(1,1,3,3)
    v_sobel=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=source_imgs1.dtype,device=device).view(1,1,3,3)
    h_sobel=torch.cat([h_sobel]*dimension[1],dim=0)
    v_sobel=torch.cat([v_sobel]*dimension[1],dim=0)
    grad_img1=torch.conv2d(source_imgs1,h_sobel,stride=1,padding=1,groups=dimension[1])**2+torch.conv2d(source_imgs1,v_sobel,stride=1,padding=1,groups=dimension[1])**2
    grad_img2 = torch.conv2d(source_imgs2, h_sobel, stride=1, padding=1, groups=dimension[1]) ** 2 + torch.conv2d(source_imgs2, v_sobel, stride=1, padding=1, groups=dimension[1]) ** 2
    five_adjacent=torch.tensor([[0,1,0],[1,1,1],[0,1,0]],dtype=source_imgs1.dtype,device=device).view(1,1,3,3)
    five_adjacent=torch.cat([five_adjacent]*dimension[1],dim=0)
    LE_img1=torch.conv2d(grad_img1,five_adjacent,stride=1,padding=1,groups=dimension[1])
    LE_img2=torch.conv2d(grad_img2,five_adjacent,stride=1,padding=1,groups=dimension[1])
    LE_img1=LE_img1/torch.max(LE_img1)
    LE_img2=LE_img2/(torch.max(LE_img2))
    E_1=E_1/torch.max(E_1)
    E_2=E_2/torch.max(E_2)
    m1=0.4*E_1+LE_img1
    m2=0.4*E_2+LE_img2
    m1=torch.exp(m1)/(torch.exp(m1)+torch.exp(m2))
    m2=torch.exp(m2)/(torch.exp(m1)+torch.exp(m2))
    fusion_1 = F.max_pool2d(m1, 3, 1, 1)
    fusion_2 = F.max_pool2d(m2, 3, 1, 1)
    m = (fusion_1> fusion_2) * 1
    res = torch.multiply(source_imgs1, m) + torch.multiply(source_imgs2, 1 - m)
    return res

