import torch
import numpy as np 
import pickle
from dataset.dataset import CompositionDataset
from torch.utils.data import DataLoader
import smact
import itertools
from smact.screening import pauling_test
from smact import neutral_ratios
from pymatgen.core.composition import Composition

def model_summary(model):
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------")
    line_new = "{:>30}  {:>20} {:>20}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #"
    )
    print(line_new)
    print("--------------------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("--------------------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

def set_random_seed(seed): 
    '''Fixes random number generator seeds for reproducibility'''
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def split_dataset(dataset, batch_size, train_ratio):
    """划分数据集"""

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split( 
                                            dataset, [train_size, test_size]
                                            )
    
    print("train length:", train_size, "test length:",test_size)
    ##Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=24, persistent_workers=True, prefetch_factor=24)
    return train_loader, test_loader


def feature2composition(feature, elements_list):
    comps = []
    max_index = feature.argmax(dim=1)
    for data in max_index:
        comp_dict = {}
        for idx, num in enumerate(data):
            if num:
                comp_dict[elements_list[idx]] = int(num)
                # print(idx, num)
        if len(comp_dict) == 0:
            comp = None
        else:
            comp = Composition.from_dict(comp_dict)
        comps.append(comp)
    return comps   


# 有效性检测
def check_valid(comp):
    ox_list = []
    paul_list = []
    elements= []
    elements_num = []
    for key, value in comp.as_dict().items():
        elem = smact.Element(key)
        elements.append(elem)
        elements_num.append([int(value)])
        ox_list.append(elem.oxidation_states)
        paul_list.append(elem.pauling_eneg)

    cn_e_list = []
    electroneg_list = []
    cn_e_electroneg_list = []
    for i in itertools.product(*ox_list):
 
        try:
            cn_e, cn_r = neutral_ratios(
                                i, elements_num
                            )
        except:
            cn_e = False
        cn_e_list.append(cn_e)
        
        try:
            electroneg_makes_sense = pauling_test(i, paul_list, elements)
        except: 
            electroneg_makes_sense = False
        electroneg_list.append(electroneg_makes_sense)
        if cn_e and electroneg_makes_sense:
            cn_e_electroneg_list.append(True)
        else:
            cn_e_electroneg_list.append(False)
    return bool(sum(cn_e_list)), bool(sum(electroneg_list)), bool(sum(cn_e_electroneg_list))