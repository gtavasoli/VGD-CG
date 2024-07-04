from DIFFUSION.utils.network import *
from torch import nn
import torch

def cosine_beta_schedule(timesteps, s=0.008, **kwargs):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class VarianceSchedule(nn.Module):
    def __init__(self, schedule_name="linear_beta_schedule", beta_start=None, beta_end=None):
        super(VarianceSchedule, self).__init__()

        self.schedule_name = schedule_name

        # 定义函数字典
        beta_schedule_dict = {'linear_beta_schedule': linear_beta_schedule,
                              'cosine_beta_schedule': cosine_beta_schedule,
                              'quadratic_beta_schedule': quadratic_beta_schedule,
                              'sigmoid_beta_schedule': sigmoid_beta_schedule}

        # 根据输入字符串选择相应的函数
        if schedule_name in beta_schedule_dict:
            self.selected_schedule = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')

        if beta_end and beta_start is None and schedule_name != "cosine_beta_schedule":
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps):
        # print(self.beta_start)
        return self.selected_schedule(timesteps=timesteps) if self.schedule_name == "cosine_beta_schedule" \
            else self.selected_schedule(timesteps=timesteps, beta_start=self.beta_start, beta_end=self.beta_end)


class DiffusionModel(nn.Module):
    def __init__(self,
                schedule_name="linear_beta_schedule",
                timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                denoise_model=None,
                data_size=None,
                elements_list=None, n_class=0, class_flag="stability", 
                icsd_label=False, semic_label=False):
        super(DiffusionModel, self).__init__()

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
        self.denoise_model = denoise_model
        # 方差生成
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
 
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
     
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond_label):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t, cond_label) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size, device, cond_label=None):
        # start from pure noise (for each example in the batch)
        img = torch.randn((batch_size, 1, *self.data_size), device=device)
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long), i, cond_label)
        return img

   
class DDIM(DiffusionModel):
    def __init__(self,
                schedule_name="linear_beta_schedule",
                timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                denoise_model=None,
                data_size=None,
                elements_list=None, n_class=0, class_flag="stability", 
                icsd_label=False, semic_label=False):
        super().__init__(
                schedule_name,
                timesteps,
                beta_start,
                beta_end,
                denoise_model,
                data_size,
                elements_list, n_class, class_flag, 
                icsd_label, semic_label)    
        

    @torch.no_grad()
    def sample(self, batch_size, device, cond_label=None, ddim_step=20, eta=0):
        # start from pure noise (for each example in the batch)
        x = torch.randn((batch_size, 1, *self.data_size), device=device)
        # new_x = torch
        error = x[19] - x[0]
        # data_list = []
        for i in range(20):
            temp = x[0] + i*0.05* error
            x[i] = temp
            # temp = temp.reshape((1, *temp.shape))
       
        time_steps = torch.arange(self.timesteps, 0, -ddim_step)-1
        if time_steps[-1] != 0:
            time_steps = torch.cat((time_steps, torch.Tensor([0]))).to(torch.long)
        loop = tqdm(range(1, len(time_steps)), desc='sampling loop time step', total=len(time_steps)-1)
        for i in loop:
            cur_t = time_steps[i-1]
            prev_t = time_steps[i]
            ab_cur = self.alphas_cumprod[cur_t]
            ab_prev = self.alphas_cumprod[prev_t]

            t_tensor = torch.full((batch_size,), cur_t, device=device, dtype=torch.long)
            eps = self.denoise_model(x, t_tensor, cond_label)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = torch.randn_like(x)

            first_term = (ab_prev / ab_cur)**0.5 * x
            second_term = ((1 - ab_prev - var)**0.5 -
                            (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
          
            third_term = var**0.5 * noise
            x = first_term + second_term + third_term
        return x
