from dataset.dataset import CompositionDataset
from torch.utils.data import DataLoader
import torch
import pickle
from utils.tool import set_random_seed, model_summary
from pymatgen.core.composition import Composition
from GAN.model.gan import DiscriminatorConv, GeneratorConv
from tqdm.auto import tqdm
from pandarallel import pandarallel
from utils.tool import *
import pandas as pd
import os

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
                print("*" * 100)
                labels = torch.zeros(batch_size, model.n_class+model.icsd_label+model.semic_label, 
                                device=device
                                )
                if model.class_label:
                    labels[:, which_class] = 1
                
                if model.icsd_label:
                    labels[:, model.n_class] = is_icsd

                if model.semic_label:
                    labels[:, model.n_class+model.semic_label] = is_semic     
   
                fake_gen = model(fake_noise, labels).cpu().squeeze().detach()
            else:
                fake_gen = model(fake_noise).cpu().squeeze().detach()
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
        data.to_csv("./output/GAN/generate_sample_cond.csv", index=False)


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

    discriminator = DiscriminatorConv()

    generator = GeneratorConv(
        train_dataset.data_size, config['z_dim'], train_dataset.elements_list, 
        config['n_class'], class_flag=config['class_flag'], icsd_label=config['icsd_label'], 
        semic_label=config['semic_label']
        )
    device = config['device']
    print('Model architectures')
    print("generator")
    model_summary(generator)
    print("discriminator")
    model_summary(discriminator)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), config['lr'], (0.1, 0.9))
    g_optimizer = torch.optim.Adam(generator.parameters(), config['lr'], (0.1, 0.9))
    print("\n\n Starting GAN training\n\n")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    generator.train()
    discriminator.train()
    epochs = config['epochs']
   
    for epoch in range(epochs): 
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (x_real, label) in loop:
            x_real = x_real.to(device)
            # cond_label = cond_label.to(device)
            # real_label = torch.ones((x_real.size(0), 1)).to(device) # 定义真实的图片label为1
            # fake_label = torch.zeros((x_real.size(0), 1)).to(device) # 定义假的图片的label为0
            label = label.to(device)
            for j in range(config['n_critic']):
                # 计算真实数据的损失
                real_out = discriminator(x_real)  # 将真实数据放入判别器中
                loss_real = -torch.mean(real_out)

                # 计算假数据的损失
                z = torch.randn(x_real.size(0), config['z_dim']).to(device)

                x_fake = generator(z, label)
            
                fake_out = discriminator(x_fake.detach())
                loss_fake = torch.mean(fake_out)
                # Clip weights of discriminator
                # for p in discriminator.parameters():
                #     p.data.clamp_(-0.01, 0.01)

                # 梯度惩罚
                alpha = torch.rand(x_real.size()[0], 1, 1, 1).expand(x_real.size()).to(device)
                one = torch.tensor([1]).to(device)
                x_hat = alpha * x_real + (one - alpha) * x_fake
                pred_hat = discriminator(x_hat)
                gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, 
                                grad_outputs=torch.ones(pred_hat.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
                ###########

                # 损失函数和优化
                d_loss = loss_real + loss_fake + config['lambda_penalty'] * gradient_penalty # 损失包括判真损失和判假损失
                d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
                d_loss.backward()  # 将误差反向传播
                d_optimizer.step()  # 更新参数
            
            z = torch.randn(x_real.size(0), config['z_dim']).to(device)
            x_fake = generator(z, label)
            fake_out = discriminator(x_fake)
            g_loss = -torch.mean(fake_out)       # 得到的假的图片与真实的图片的label的loss
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(**{'D_loss': d_loss.item(), 'G_loss': g_loss.item()})
    
        # if (epoch+1) % 5 == 0:
        #     generate_samples(model_path='./saved_model/GAN/GAN_Conv.pt', batch_size=10000, epochs=1)
    torch.save(generator, os.path.join("./saved_model/GAN", config['model_name'])) 
    print("\nFinished GAN training")



if __name__ == "__main__":
    import time
    config = {'seed': 6, 'z_dim': 128, "n_critic": 5, "semic_label": True,
               'batch_size': 4000, 'lambda_penalty': 10,
          'lr': 0.001, 'epochs': 20, "n_class": 2, "class_flag": "stability", "icsd_label": True,
            'model_name': 'GAN_Conv.pt'
            }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is available:"+str(torch.cuda.is_available())+", Quantity: "+str(torch.cuda.device_count())+'\n')
    print(f"Running on {device}")
    set_random_seed(config['seed'])
    config['device'] = device
    # time_1 = time.time()
    # train(config)
    time_2 = time.time()
    generate_samples(model_path=f"./saved_model/GAN/{config['model_name']}", batch_size=10000, epochs=1, is_icsd=True,
        is_semic=True)
    time_3 = time.time()
    # print(time_2 - time_1)
    print(time_3 - time_2)