import torch

class AE_Linear(torch.nn.Module):
    def __init__(self, data_size, hidden_dims, elements_list):
        super().__init__()
        # 定义编码器
        modules = []
        hidden_dims.insert(0, data_size[0]*data_size[1]) # 将数据的channel插入
        self.data_size = data_size
        self.hidden_dims = hidden_dims
        self.elements_list = elements_list

        for i in range(len(hidden_dims)-1):
            modules.append(
            torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]),
            torch.nn.BatchNorm1d(hidden_dims[i+1]),
            torch.nn.ReLU())
            )
        self.encoder = torch.nn.Sequential(*modules)

        modules = []
        for i in reversed(range(1, len(hidden_dims))):
            modules.append(
            torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], hidden_dims[i-1]),
            torch.nn.BatchNorm1d(hidden_dims[i-1]),
            torch.nn.ReLU())
            )
        self.decoder = torch.nn.Sequential(*modules)

    def encode(self, x):
        # Encode the input given a class label and return mean and log variance
        x = x.view(x.shape[0], -1)
        out = self.encoder(x) 
        return out

    def decode(self, z):
        # Decode a sample from the latent space given a class label

        out = self.decoder(z).view(-1, *self.data_size)
        return torch.sigmoid(out)
        # return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class AE_Conv(torch.nn.Module):
    def __init__(self, data_size, z_dim, hidden_dims, elements_list):
        super().__init__()
        # 定义编码器
        modules = []
        # hidden_dims.insert(0, 1) # 将数据的channel插入
        self.data_size = data_size
        self.hidden_dims = hidden_dims
        self.elements_list = elements_list
        self.z_dim = z_dim
        modules = []
        
        modules += self.encode_block(1, 32, kernel_size=(3, 4), stride=(1, 2)) # 9, 42
        modules += self.encode_block(32, 64, kernel_size=(3, 4), stride=(1, 2))# 9, 21
        modules += self.encode_block(64, 128, kernel_size=(3, 3), stride=(1, 2))# 9, 11
        modules += self.encode_block(128, 256, kernel_size=(3, 3), stride=(2, 2))  # 5, 6
        self.encoder = torch.nn.Sequential(*modules)
        
        self.encoder_linear = torch.nn.Linear(5*6*256, z_dim)
        self.decoder_linear = torch.nn.Linear(z_dim, 5*6*256)
        modules = []
        modules += self.decode_block(256, 128, kernel_size=(3, 3), stride=(2, 2))
        modules += self.decode_block(128, 64, kernel_size=(3, 3), stride=(1, 2))
        modules += self.decode_block(64, 32, kernel_size=(3, 4), stride=(1, 2))
        modules += self.decode_block(32, 1, kernel_size=(3, 4), stride=(1, 2))

        self.decoder = torch.nn.Sequential(*modules)
        

    def encode_block(self, in_dim, out_dim, kernel_size, stride):
        data = [
        torch.nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
        torch.nn.BatchNorm2d(out_dim),
        torch.nn.ReLU()]
        
        return data
        
    def decode_block(self, in_dim, out_dim, kernel_size, stride):
        data = [
        torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
        torch.nn.BatchNorm2d(out_dim),
        torch.nn.ReLU()]
        return data
        
    def encode(self, x):
        # Encode the input given a class label and return mean and log variance
        out = self.encoder(x) 
        out = self.encoder_linear(out.view(out.shape[0], -1))
        return out
     

    def decode(self, z):
        # Decode a sample from the latent space given a class label
        out = self.decoder_linear(z)
        out = self.decoder(out.view(z.shape[0], -1, 5, 6))
        # out = self.decoder(z)
        return torch.sigmoid(out)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
