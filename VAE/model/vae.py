import torch



class VAE_Linear(torch.nn.Module):
    def __init__(self, data_size, z_dim, hidden_dims, elements_list):
        super().__init__()
        # 定义编码器
        modules = []
        hidden_dims.insert(0, data_size[0]*data_size[1]) # 将数据的channel插入
        self.data_size = data_size
        self.hidden_dims = hidden_dims
        self.elements_list = elements_list
        self.z_dim = z_dim
        for i in range(len(hidden_dims)-1):
            modules.append(
            torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]),
            torch.nn.BatchNorm1d(hidden_dims[i+1]),
            torch.nn.LeakyReLU())
            )
        self.encoder = torch.nn.Sequential(*modules)

        self.mean_linear = torch.nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_linear = torch.nn.Linear(hidden_dims[-1], z_dim)

        self.decoder_projection = torch.nn.Linear(
            z_dim, hidden_dims[-1])
        modules = []
        for i in reversed(range(1, len(hidden_dims))):
            modules.append(
            torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], hidden_dims[i-1]),
            torch.nn.BatchNorm1d(hidden_dims[i-1]),
            torch.nn.LeakyReLU())
            )
        self.decoder = torch.nn.Sequential(*modules)

    def encode(self, x):
        # Encode the input given a class label and return mean and log variance
        x = x.view(x.shape[0], -1)
        out = self.encoder(x) 
        return self.mean_linear(out), self.logvar_linear(out)

    def sampling(self, z_mean, z_logvar):
        # Sample from latent space using reparametrization trick
        epsilon = torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        # Decode a sample from the latent space given a class label
        out = self.decoder_projection(z)
        out = self.decoder(out).view(-1, *self.data_size)
        return torch.sigmoid(out)

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)  
        return self.decode(z), z_mean, z_logvar


class VAE_Conv(torch.nn.Module):
    def __init__(self, data_size, z_dim, elements_list, n_class=0, class_flag="stability", 
                icsd_label=False, semic_label=False
            ):
        super().__init__()
        # Defining the encoder
        modules = []
        # hidden_dims.insert(0, 1) # Insert the data channel
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
        modules = []
        modules += self.encode_block(1, 32, kernel_size=(3, 3), stride=(1, 1)) # 9, 84
        modules += self.encode_block(32, 32, kernel_size=(3, 4), stride=(1, 2)) # 9, 42
        modules += self.encode_block(32, 64, kernel_size=(3, 3), stride=(1, 1))# 9, 42
        modules += self.encode_block(64, 64, kernel_size=(3, 4), stride=(1, 2))# 9, 21
        modules += self.encode_block(64, 128, kernel_size=(3, 3), stride=(1, 1))# 9, 21
        modules += self.encode_block(128, 128, kernel_size=(3, 3), stride=(1, 2))# 9, 11
        modules += self.encode_block(128, 256, kernel_size=(3, 3), stride=(2, 2))  # 5, 6
        self.encoder = torch.nn.Sequential(*modules)
        
        self.mean_linear = torch.nn.Linear(5*6*128, z_dim)
        self.logvar_linear = torch.nn.Linear(5*6*128, z_dim)
        self.decoder_linear = torch.nn.Linear(z_dim+n_class+icsd_label+semic_label, 5*3*256)
        modules = []
        modules += self.decode_block(256, 128, kernel_size=(3, 3), stride=(2, 2))
        modules += self.decode_block(128, 128, kernel_size=(3, 3), stride=(1, 2))
        modules += self.decode_block(128, 64, kernel_size=(3, 3), stride=(1, 1))
        modules += self.decode_block(64, 64, kernel_size=(3, 4), stride=(1, 2))
        modules += self.decode_block(64, 32, kernel_size=(3, 3), stride=(1, 1))
        modules += self.decode_block(32, 32, kernel_size=(3, 4), stride=(1, 2), output_padding=(0, 1))
        modules += self.decode_block(32, 1, kernel_size=(3, 3), stride=(1, 1))

        self.decoder = torch.nn.Sequential(*modules)

    def sampling(self, z_mean, z_logvar):
        # Sample from latent space using reparametrization trick
        epsilon = torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def encode_block(self, in_dim, out_dim, kernel_size, stride, padding=1):
        data = [
        torch.nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(out_dim),
        torch.nn.ReLU()]
        return data
        
    def decode_block(self, in_dim, out_dim, kernel_size, stride, padding=1, output_padding=0):
        data = [
        torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, 
                                 padding=padding, output_padding=output_padding),
        torch.nn.BatchNorm2d(out_dim),
        torch.nn.ReLU()]
        return data
        
    def encode(self, x):
        # Encode the input given a class label and return mean and log variance
        out = self.encoder(x) 
        # print("Encoder output shape:", out.shape)
        return self.mean_linear(out.view(out.shape[0], -1)), self.logvar_linear(out.view(out.shape[0], -1))
     

    def decode(self, z, c=None):
        # Decode a sample from the latent space given a class label
     
        if self.class_label or self.icsd_label or self.semic_label:
            z_c = torch.concat((z, c), dim=1)
        else:
            z_c = z
        out = self.decoder_linear(z_c)
        out = self.decoder(out.view(z.shape[0], -1, 5, 3))
        # out = self.decoder(z)
        return torch.sigmoid(out)
    
    def forward(self, x, c=None):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)  
        return self.decode(z, c), z_mean, z_logvar
    

    
class CVAE_Conv(torch.nn.Module):
    def __init__(self, data_size, z_dim, n_class, hidden_dims, elements_list):
        super().__init__()
        # 定义编码器
        modules = []
        hidden_dims.insert(0, 1) # 将数据的channel插入
        self.data_size = data_size
        self.hidden_dims = hidden_dims
        self.elements_list = elements_list
        self.n_class = n_class
        self.z_dim = z_dim
        for i in range(len(hidden_dims)-1):
            modules.append(
            torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[i+1]),
            torch.nn.LeakyReLU())
            )
        self.encoder = torch.nn.Sequential(*modules)

        self.mean_linear = torch.nn.Linear(hidden_dims[-1]*data_size[0]*data_size[1], z_dim)
        self.logvar_linear = torch.nn.Linear(hidden_dims[-1]*data_size[0]*data_size[1], z_dim)

        self.decoder_projection = torch.nn.Linear(
            z_dim+n_class+1, hidden_dims[-1]*data_size[0]*data_size[1])
        modules = []
        for i in reversed(range(1, len(hidden_dims))):
            modules.append(
            torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(hidden_dims[i-1]),
            torch.nn.LeakyReLU())
            )
        self.decoder = torch.nn.Sequential(*modules)

    def encode(self, x):
        # Encode the input given a class label and return mean and log variance
        
        out = self.encoder(x) 
        return self.mean_linear(out.view(out.shape[0], -1)), self.logvar_linear(out.view(out.shape[0], -1))

    def sampling(self, z_mean, z_logvar):
        # Sample from latent space using reparametrization trick
        epsilon = torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        # Decode a sample from the latent space given a class label
        out = self.decoder_projection(z)
        out = self.decoder(out.view(-1, self.hidden_dims[-1], *self.data_size))
        return torch.sigmoid(out)

    def forward(self, x, c):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        z_c = torch.cat((z, c), dim=-1)      
        return self.decode(z_c), z_mean, z_logvar
