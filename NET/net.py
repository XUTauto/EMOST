import torch
from torch import nn
from STB import TransformerBlock
from EMO import EMO
class ChannelATT(nn.Module):
	def __init__(self, channels, reduction=16):
		super(ChannelATT, self).__init__()
		self.avg = nn.AdaptiveAvgPool2d(1)
		self.fully1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
		self.relu = nn.ReLU(inplace=True)
		self.fully2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
		self.sg = nn.Sigmoid()
	def forward(self, input):
		x = self.avg(input)
		x = self.fully1(x)
		x = self.relu(x)
		x = self.fully2(x)
		x = self.sg(x)
		return input * x
class SpatialATT(nn.Module):
	def __init__(self, channels, reduction=16):
		super(SpatialATT, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(channels // reduction, 1, kernel_size=3, padding=1)
		self.sg = nn.Sigmoid()
	def forward(self, input):
		x = self.conv1(input)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.sg(x)
		return input * x
LayerNorm_type = 'WithBias'
num_blocks = [4, 6, 6, 8]
heads = [1, 2, 4, 8]
class MODEL_EMO(nn.Module):
	def __init__(self,out_channels=64, groups=1):
		super(MODEL_EMO, self).__init__()
		self.conv0 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
		self.conv1 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
		self.conv2 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
		self.conv3 = nn.Conv2d(192, 64, (3, 3), (1, 1), (1, 1))
		self.conv4 = nn.Conv2d(256, 64, (1, 1), (1, 1), (0, 0))
		self.conv5 = nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0))
		self.conv6 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
		self.norm16 = nn.BatchNorm2d(16)
		self.norm64 = nn.BatchNorm2d(64)
		bottleneck_planes = groups * out_channels
		self.lu = nn.ReLU()
		self.tanh=nn.Tanh()
		self.Conv3x3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
		self.emo = EMO(dim_in=64,
					   depths=[1, 2, 4, 2], stem_dim=64, embed_dims=[64, 128, 256, 512], exp_ratios=[4., 4., 4., 4.],
					   norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
					   dw_kss=[3, 3, 5, 5], se_ratios=[0.0, 0.0, 0.0, 0.0], dim_heads=[32, 32, 32, 32],
					   window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True], qkv_bias=True)
		self.se = ChannelATT(bottleneck_planes)
		self.sa = SpatialATT(bottleneck_planes)
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.down = nn.AvgPool2d(2, 2)
	def forward(self, input):
		x = input
		x = self.conv0(x)
		out1 = self.conv1(x)
		out2 = self.norm64(out1)
		out2 = self.lu(out2)
		out20 = torch.cat([x, out2], dim=1)
		out3 = self.conv2(out20)
		out3 = self.norm64(out3)
		out3 = self.lu(out3)
		out30 = torch.cat([x, out2, out3], dim=1)
		out4 = self.conv3(out30)
		out4 = self.norm64(out4)
		out4 = self.lu(out4)
		out40 = torch.cat([x, out2, out3, out4], dim=1)
		output1 = self.conv4(out40)
		l5 = torch.cat([x, out1], dim=1)
		l5=self.conv5(l5)
		l51=self.lu(self.norm64(self.conv6(self.lu(self.norm64(self.conv6(self.lu(self.norm64(self.conv6(l5)))))))))
		l52=self.lu(self.conv1(self.lu(self.norm64(self.conv6(l5)))))
		l5=torch.cat([l51,l52], dim=1)
		l5=self.conv5(l5)

		l51=self.conv6(self.conv6(self.conv6(l5)))
		l52=self.conv1(self.conv6(l5))
		l5=torch.cat([l51,l52], dim=1)
		l5=self.conv5(l5)

		l51=self.conv6(self.conv6(self.conv6(l5)))
		l52=self.conv1(self.conv6(l5))
		l5=torch.cat([l51,l52], dim=1)
		l5=self.conv5(l5)

		l5 = self.emo(l5)
		output1 = torch.mul(l5, output1) + output1
		l6_up=self.se(output1)
		l6_down = self.sa(output1)
		tol=torch.cat([l6_up, l6_down], dim=1)
		tol=self.conv5(tol)+output1
		tol=self.lu(tol)
		output1 = self.emo(tol)
		return output1


class ConvLayer(nn.Module):
	def __init__(self, in_c, out_c, kernel_size, stride, is_last=False):
		super(ConvLayer, self).__init__()
		padding_size = int(kernel_size // 2)
		self.padding = nn.ReflectionPad2d(padding_size)
		self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride)
		self.prelu = nn.PReLU()
		self.is_last = is_last
	def forward(self, x):
		x = self.padding(x)
		x = self.conv(x)
		if self.is_last == False:
			x = self.prelu(x)
		return x

class MODEL_DRS(nn.Module):
	def __init__(self):
		super(MODEL_DRS, self).__init__()
		self.conv5 = nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0))
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.down = nn.AvgPool2d(2, 2)
		self.c1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
		self.c2 = nn.Conv2d(192, 64, (1, 1), (1, 1), (0, 0))
		self.Conv3x3 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))
		self.emo = EMO(dim_in=64,
		               depths=[1, 2, 4, 2], stem_dim=64, embed_dims=[64, 128, 256, 512], exp_ratios=[4., 4., 4., 4.],
		               norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
		               dw_kss=[3, 3, 5, 5], se_ratios=[0.0, 0.0, 0.0, 0.0], dim_heads=[32, 32, 32, 32],
		               window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True], qkv_bias=True)  # stem_dim代表输出维度
		self.drs = nn.Sequential(*[TransformerBlock(
			dim=64,
			num_heads=heads[0],
			ffn_expansion_factor=2.66,
			bias=False,
			LayerNorm_type=LayerNorm_type)
			for i in range(num_blocks[0])])
		self.drs1 = nn.Sequential(*[TransformerBlock(
			dim=64,
			num_heads=heads[0],
			ffn_expansion_factor=2.66,
			bias=False,
			LayerNorm_type=LayerNorm_type)
			for i in range(num_blocks[0])])
		self.drs2 = nn.Sequential(*[TransformerBlock(
			dim=128,
			num_heads=heads[0],
			ffn_expansion_factor=2.66,
			bias=False,
			LayerNorm_type=LayerNorm_type)
			for i in range(num_blocks[0])])
		self.drs3 = nn.Sequential(*[TransformerBlock(
			dim=192,
			num_heads=heads[0],
			ffn_expansion_factor=2.66,
			bias=False,
			LayerNorm_type=LayerNorm_type)
			for i in range(num_blocks[0])])

	def forward(self, input):
		x = self.Conv3x3(input)
		l1 = self.c1(x)
		l2_0 = self.down(x)
		l2 = self.c1(self.c1(l2_0)) + l2_0
		l2 = self.up(l2)
		l3_0 = self.down(self.down(x))
		l3 = self.c1(self.c1(self.c1(l3_0))) + l3_0
		l3 = self.up(self.up(l3))
		l4 = torch.cat([l2, l3], dim=1)
		l4 = self.conv5(l4)
		l5 = torch.mul(l4, l1)
		l6 = l5 + l1 + l2 + l3
		l6_2 = self.drs1(l6)
		l6_2 = torch.cat([l6, l6_2], dim=1)
		l6_3 = self.drs2(l6_2)
		l6_3 = torch.cat([l6, l6_3], dim=1)
		l6_4 = self.drs3(l6_3)
		l61 = self.c2(l6_4)
		return l61





class MODEL_ADD(nn.Module):
	def __init__(self):
		super(MODEL_ADD, self).__init__()
		self.ONE = MODEL_EMO()
		self.TWO = MODEL_DRS()

	def forward(self, input1):
		UP = self.ONE(input1)
		DOWN = self.TWO(input1)
		out =  UP+DOWN
		return out




class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.conv1 = nn.Conv2d(64, 64,(3, 3), (1, 1), (1, 1))
		self.conv2 = nn.Conv2d(64, 64,  (3, 3), (1, 1), (1, 1))
		self.conv3 = nn.Conv2d(128, 64,  (3, 3), (1, 1), (1, 1))
		self.conv4 = nn.Conv2d(128, 64,(3, 3), (1, 1), (1, 1))
		self.conv5 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))
		self.relu = nn.ReLU()
	def forward(self, img):
		db1 = self.conv1(img)
		db2 = self.conv2(db1)
		db3 = self.conv3(torch.cat([db1, db2], dim=1))
		db4 = self.conv4(torch.cat([db2, db3], dim=1))
		output = self.conv5(db4)
		output = self.relu(output)
		return output


class TOTAL(nn.Module):
	def __init__(self):
		super(TOTAL, self).__init__()
		self.Extraction = MODEL_ADD()
		self.Reconstruction = Decoder()

	def forward(self, input1):
		fu = self.Extraction(input1)
		out = self.Reconstruction(fu)
		return out
