import torch


class DiscriminatorConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
            # 定义编码器
        modules = []
        # modules += self.encode_block(1, 32, kernel_size=(3, 3), stride=(1, 1)) # 9, 84
        # modules += self.encode_block(32, 32, kernel_size=(3, 4), stride=(1, 2)) # 9, 42
        # modules += self.encode_block(32, 64, kernel_size=(3, 3), stride=(1, 1))# 9, 42
        # modules += self.encode_block(64, 64, kernel_size=(3, 4), stride=(1, 2))# 9, 21
        # modules += self.encode_block(64, 128, kernel_size=(3, 3), stride=(1, 1))# 9, 21
        # modules += self.encode_block(128, 128, kernel_size=(3, 3), stride=(1, 2))# 9, 11
        # modules += self.encode_block(128, 256, kernel_size=(3, 3), stride=(2, 2))  # 5, 6
        # self.disc = torch.nn.Sequential(*modules)
        # self.out_linear = torch.nn.Linear(5*6*256, 1)
        chanel = 32
        modules += self.encode_block(1, chanel, kernel_size=(3, 3), stride=(1, 1)) # 9, 84
        modules += self.encode_block(chanel, chanel, kernel_size=(3, 4), stride=(1, 2)) # 9, 42
        modules += self.encode_block(chanel, chanel*2, kernel_size=(3, 3), stride=(1, 1))# 9, 42
        modules += self.encode_block(chanel*2, chanel*2, kernel_size=(3, 4), stride=(1, 2))# 9, 21
        modules += self.encode_block(chanel*2, chanel*4, kernel_size=(3, 3), stride=(1, 1))# 9, 21
        modules += self.encode_block(chanel*4, chanel*4, kernel_size=(3, 3), stride=(1, 2))# 9, 11
        modules += self.encode_block(chanel*4, chanel*8, kernel_size=(3, 3), stride=(2, 2))  # 5, 6
        self.disc = torch.nn.Sequential(*modules)
        self.out_linear = torch.nn.Linear(5*6*chanel*8, 1)

    
    def encode_block(self, in_dim, out_dim, kernel_size, stride):
        data = [
        torch.nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
        torch.nn.InstanceNorm2d(out_dim),
        torch.nn.ReLU()]
        return data
    
    def forward(self, x):
        out = self.disc(x)
        out = self.out_linear(out.view(out.shape[0], -1))
        return out

class GeneratorConv(torch.nn.Module):
    
    def __init__(self, data_size, z_dim, elements_list, n_class=0, class_flag="stability", 
                icsd_label=False, semic_label=False
            ):
        super().__init__()
        modules = []
        # hidden_dims.insert(0, 1) # 将数据的channel插入
        self.data_size = data_size
        self.n_class = n_class
        self.icsd_label = icsd_label
        self.class_flag = class_flag
        self.elements_list = elements_list
        self.semic_label = semic_label
        if self.n_class >= 2 :
            self.class_label = True
        else:
            self.class_label = False
        self.z_dim = z_dim
        chanel = 32
        self.input_linear = torch.nn.Linear(z_dim+n_class+icsd_label+semic_label, 5*6*chanel*8)
        modules = []
        
        modules += self.decode_block(chanel*8, chanel*4, kernel_size=(3, 3), stride=(2, 2))
        modules += self.decode_block(chanel*4, chanel*4, kernel_size=(3, 3), stride=(1, 2))
        modules += self.decode_block(chanel*4, chanel*2, kernel_size=(3, 3), stride=(1, 1))
        modules += self.decode_block(chanel*2, chanel*2, kernel_size=(3, 4), stride=(1, 2))
        modules += self.decode_block(chanel*2, chanel, kernel_size=(3, 3), stride=(1, 1))
        modules += self.decode_block(chanel, chanel, kernel_size=(3, 4), stride=(1, 2))
        modules += self.decode_block(chanel, 1, kernel_size=(3, 3), stride=(1, 1))
        self.gener = torch.nn.Sequential(*modules)
    
    def decode_block(self, in_dim, out_dim, kernel_size, stride):
        data = [
        torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
        torch.nn.InstanceNorm2d(out_dim),
        torch.nn.ReLU()]
        return data
        
    def forward(self, z, c=None):

        if self.class_label or self.icsd_label or self.semic_label:
            z_c = torch.concat((z, c), dim=1)
        else:
            z_c = z
        out = self.input_linear(z_c)
        out = self.gener(out.view(z.shape[0], -1, 5, 6))
        return torch.tanh(out)


