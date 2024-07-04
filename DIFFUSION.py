
from torch.utils.data import DataLoader
import torch
from DIFFUSION.model.unet import Unet
from DIFFUSION.model.diffusion import DiffusionModel, DDIM
import os
import yaml
from torch.optim import Adam
from tqdm.auto import tqdm
from utils.tool import set_random_seed, model_summary
from dataset.dataset import CompositionDataset
from torch.utils.data import DataLoader
import pickle
from pymatgen.core.composition import Composition
from utils.tool import *
from pandarallel import pandarallel
import pandas as pd 

def generate_samples(
        model_path, 
        epochs=1, 
        batch_size=2500, 
        which_class=0, 
        device="cuda",
        is_icsd=False,
        is_semic=False
        ):
        """ Helper function to iteratively generate valid samples """  
        comp_list = []
        model = torch.load(model_path)
        model.to(device)
        model.eval()
        
        for i in range(epochs):
            # samples = model.sample(batch_size, device).cpu().squeeze().detach()
            # 条件编码

            
            if model.class_label or model.icsd_label or model.semic_label:
                
                labels = torch.zeros(batch_size, model.n_class+model.icsd_label+model.semic_label, 
                                device=device
                                )
                if model.class_label:
                    labels[:, which_class] = 1
                
                if model.icsd_label:
                    labels[:, model.n_class] = is_icsd

                if model.semic_label:
                    labels[:, model.n_class+model.semic_label] = is_semic     
   
                fake_gen = model.sample(batch_size, device, labels).cpu().squeeze().detach()
            else:
                fake_gen = model.sample(batch_size, device).cpu().squeeze().detach()
            comp_list += feature2composition(fake_gen, model.elements_list)
        data = pd.DataFrame({'composition': comp_list})

        data = data.dropna()
        print("删除None后：{}/{}".format(data.shape[0], len(comp_list)))
        # data = data.drop_duplicates()
        # print("删除重复后：{}/{}".format(data.shape[0], len(comp_list)))
        

        pandarallel.initialize(progress_bar=False, nb_workers=48)

        data['element_num'] = data['composition'].parallel_apply(lambda comp: len(comp.elements))
        
        print(data['element_num'].value_counts())
     
        data = data[data['element_num']>1]
        data = data[data['element_num']<6]
        print("2~5之间的化合物：{}/{}".format(data.shape[0], len(comp_list)))

        temp = data['composition'].parallel_apply(check_valid)
        data['charge neutrality'] = temp.parallel_apply(lambda comp: comp[0])
        data['electronegativity balance'] = temp.parallel_apply(lambda comp: comp[1])
        data['valid'] = temp.parallel_apply(lambda comp: comp[2])
        
        print("charge neutrality: {}/{} {}".format(data['charge neutrality'].sum(), data.shape[0], 
                                        data['charge neutrality'].sum()/data.shape[0])
                                        )
        print("electronegativity balance: {}/{} {}".format(data['electronegativity balance'].sum(), 
                                        data.shape[0], data['electronegativity balance'].sum()/data.shape[0])
                                        )
        print("valid: {}/{} {}".format(data['valid'].sum(), data.shape[0], data['valid'].sum()/data.shape[0]))
        data['composition'] = data['composition'].parallel_apply(lambda comp: comp.formula)
        data.to_csv("./output/DIFFUSION/generate_sample_cond.csv", index=False)






def train(config): 
    train_dataset = CompositionDataset(target_name="train.csv", n_class=config['n_class'], 
                                       class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                                       semic_label=config['semic_label']
                                       )
    test_dataset = CompositionDataset(target_name="test.csv", n_class=config['n_class'], 
                                      class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                                      semic_label=config['semic_label']
                                      )
    print("train length:", len(train_dataset), "test length:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24
                              ) 
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                             pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24
                             )
    torch.save((train_loader, test_loader), "./data/train_test_loader.pt")
    epochs = config['epochs']
    device = config['device']
    timesteps = config['timesteps']
    beta_start = config['beta_start']
    beta_end = config['beta_end']

    dim_mults = (1, 2, 4)
    denoise_model = Unet(
        dim=32,
        channels=1,
        dim_mults=dim_mults,
        c_dim=train_dataset.n_class+train_dataset.icsd_label+train_dataset.semic_label
    ).to(device)

    model = DDIM(schedule_name=config['schedule_name'],
                        timesteps = timesteps,
                        beta_start = beta_start,
                        beta_end = beta_end,
                        denoise_model=denoise_model,
                        data_size=train_dataset.data_size,
                        elements_list=train_dataset.elements_list, 
                        n_class=config['n_class'], class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                        semic_label=config['semic_label']
                    )
    
    print('Model architectures')
    model_summary(model) 

    optimizer = Adam(model.parameters(), lr=config['lr'])
    
    print("\n\n Starting DDPM training\n\n")
   
    for i in range(epochs):
        
        losses = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (x_real, labels) in loop:
            x_real = x_real.to(device)
            batch_size = x_real.shape[0]
            labels = labels.to(device)
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            # loss = model(mode="train", x_start=x_real, t=t, loss_type="huber")
            noise = torch.randn_like(x_real)
            x_noisy = model.q_sample(x_start=x_real, t=t, noise=noise)  # 添加噪声之后的
            
            if model.class_label or model.icsd_label or model.semic_label:
                predicted_noise = model.denoise_model(x_noisy, t, labels)
            else:
                predicted_noise = model.denoise_model(x_noisy, t)
            
            loss = torch.nn.functional.smooth_l1_loss(noise, predicted_noise)
            
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # 更新信息
            loop.set_description(f'Epoch [{i}/{epochs}]')
            loop.set_postfix(loss=loss.item())
        
        # if "model_save_path" in kwargs.keys():
        #     self.save_best_model(model=model, path=kwargs["model_save_path"])
    torch.save(model, os.path.join("./saved_model/DIFFUSION", config['model_name']))  
    print("\nFinished DDPM training")



if __name__ == "__main__":
    import time
    config = {'seed': 6, "beta_end": 0.02, "timesteps": 500, "semic_label": True,
               'batch_size': 256, 'beta_start': 0.0001, "schedule_name": "linear_beta_schedule",
          'lr': 0.0005, 'epochs': 20, "n_class": 2, "class_flag": "stability", "icsd_label": True,
            'model_name': 'DIFFUSION.pt'}
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(config['seed'])
    # device = 'cpu'
    config['device'] = device
    # time1 = time.time()
    # train(config)
    time2 = time.time()
    generate_samples(model_path=f"./saved_model/DIFFUSION/{config['model_name']}", batch_size=100, epochs=1,
                    is_icsd=True, is_semic=True
                    )
    time3 = time.time()
    # print(time2-time1)
    print(time3-time2)