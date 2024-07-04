from dataset.dataset import CompositionDataset
from torch.utils.data import DataLoader
from VAE.model.vae import VAE_Conv
import torch
import pandas as pd 
import pickle
from pymatgen.core.composition import Composition
from utils.tool import *
from pandarallel import pandarallel
from tqdm.auto import tqdm
import os


def reconstruct_samples(model, data_loader=None):
    model.eval()
    comp_list = [[], []]
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (x, label) in loop:
        x = x.to(device)
        # label = label.to(device)
        x_hat, _, _ = model(x)
        x = x.squeeze()
        x_hat = x_hat.squeeze()
        for i, d in enumerate([x, x_hat]):
            comp_list[i] += feature2composition(d, model.elements_list)
    data = pd.DataFrame({'real': comp_list[0], 'fake': comp_list[1]})
    same_num = sum(data['real'] == data['fake'])
    print("reconstruction: {}/{} {}".format(same_num, data.shape[0], same_num/data.shape[0]))
 

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
            fake_noise = torch.randn(batch_size, model.z_dim).float().to(device)
            # 条件编码

            
            if model.class_label or model.icsd_label or model.semic_label:
                # print("*" * 100)
                labels = torch.zeros(batch_size, model.n_class+model.icsd_label+model.semic_label, 
                                device=device
                                )
                if model.class_label:
                    labels[:, which_class] = 1
                
                if model.icsd_label:
                    labels[:, model.n_class] = is_icsd

                if model.semic_label:
                    labels[:, model.n_class+model.semic_label] = is_semic     
   
                fake_gen = model.decode(fake_noise, labels).cpu().squeeze().detach()
            else:
                fake_gen = model.decode(fake_noise).cpu().squeeze().detach()
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
        data.to_csv("./output/VAE/generate_sample_cond.csv", index=False)
        
        

def loss_func(x, x_hat, mean, logvar, w_kl=1):
    """weighted bce loss"""
    x = x.view(x.shape[0], -1)
    x_hat = x_hat.view(x_hat.shape[0], -1)
    weight = torch.where(x==0, 1/9, 8/9)
    bce = torch.nn.BCELoss(weight=weight, reduction='none')
    bce_loss = torch.mean(torch.sum(bce(x_hat, x), dim=1), dim=0)
    kl_loss = torch.mean(-0.5*torch.sum(1+logvar-mean.pow(2)-logvar.exp(), dim=1), dim=0)

    return bce_loss + w_kl*kl_loss

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

    model = VAE_Conv(train_dataset.data_size, config['z_dim'], train_dataset.elements_list, 
                     config['n_class'], class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
                     semic_label=config['semic_label']
                     )
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
    device = config['device']
    print('Model architectures')
    model_summary(model)
    print("\n\n Starting VAE training\n\n")

    model = model.to(device)
    model.train()
    epochs = config['epochs']
 
    for epoch in range(epochs): 
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x, label) in loop:
            x = x.to(device)
            label = label.to(device)
            x_hat, mean, logvar = model(x, label)

            loss = loss_func(x, x_hat, mean, logvar, config['w_kl'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item())

# if ((epoch+1) % 50 == 0):
    # print('\n计算重建损失率')
    # # print("train_loader:")
    # # reconstruct_samples(model, train_loader)
    # print("test_loader:")
    # reconstruct_samples(model, test_loader)
      
    torch.save(model, os.path.join("./saved_model/VAE", config['model_name'])) 
    print("\nFinished VAE training")
    # Saving models
   
    



if __name__ == "__main__":
    import time
    config = {'seed': 6, 'z_dim': 128, 'w_kl': 0.01, 'batch_size': 5120, "semic_label": True,
          'lr': 0.001, 'epochs': 20, "n_class": 2, "class_flag": "stability", "icsd_label": True,
            'model_name': 'VAE_Conv_cond.pt'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is available:"+str(torch.cuda.is_available())+", Quantity: "+str(torch.cuda.device_count())+'\n')
    print(f"Running on {device}")
    set_random_seed(config['seed'])
    config['device'] = device
    time_1 = time.time()
    train(config)
    time_2 = time.time()
    generate_samples(model_path=f"./saved_model/VAE/{config['model_name']}", batch_size=10000, epochs=130, 
                    is_icsd=True, is_semic=True
                    )
    time_3 = time.time()
    print(time_2 - time_1)
    print(time_3 - time_2)