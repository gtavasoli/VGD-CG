import torch
from torch.utils.data import Dataset
import pandas as pd
from pymatgen.core.composition import Composition
import numpy as np
import os
from pandarallel import pandarallel

NUM_WORKERS= os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1


class CompositionDataset(Dataset):
    def __init__(self, data_path='./data/', target_name='data.csv', n_class=5, class_flag='stability', icsd_label=False, semic_label=False, max_atom_num_per_element=8) -> None:
        super().__init__()
        self.data_path = data_path
        raw_data = pd.read_csv(os.path.join(self.data_path, target_name), 
                               dtype={
                                   'band_gap': 'float'
                                   })
        
        raw_data = raw_data.iloc[:int(raw_data.shape[0]/100), :] # DON'T FORGET TO COMMENT IT
        
        self.n_class = n_class
        self.icsd_label = icsd_label
        self.class_flag = class_flag
        self.semic_label = semic_label

        pandarallel.initialize(progress_bar=False, nb_workers=NUM_WORKERS)
        # Convert to components
        raw_data['composition'] = raw_data['composition'].parallel_apply(lambda comp: Composition(comp))
        self.elements_list = self.get_elements_list(raw_data)
        data_file = os.path.join(self.data_path, target_name.split('.')[0]+".pt")
        if os.path.exists(data_file):
            print("The dataset already exists, loading data!")
            self.features, self.labels, self.cond_bin = torch.load(data_file)
        else:
            print('Processing data!')
            # Get the maximum number of atoms
            # max_atom_num_per_element = int(raw_data['composition'].parallel_apply(lambda comp: max(comp.as_dict().values())).max())
            self.features = raw_data['composition'].parallel_apply(self.get_features, args=(max_atom_num_per_element,))
            self.labels, self.cond_bin = self.get_labels(raw_data)
            torch.save((self.features, self.labels, self.cond_bin), data_file)
        self.data_size = self.features.iloc[0].shape
        
    def get_labels(self, data):
        cond_bin = None
        labels = torch.zeros((data.shape[0], 1)) 
        data = data.sort_values(by=[self.class_flag])

        self.class_label = False
        
        if self.n_class >= 2:
            self.class_label = True
            class_labels = torch.zeros((data.shape[0], self.n_class)) 
            num = int(data.shape[0]/(self.n_class)) 
            cond_bin = [0]
            for i in range(self.n_class):
                if i == self.n_class-1:
                    class_labels[i*num:, i] = 1
                    cond_bin.append(data[self.class_flag].iloc[-1])
                else:
                    class_labels[i*num:(i+1)*num, i] = 1
                    cond_bin.append(data[self.class_flag].iloc[(i+1)*num])
            labels = torch.concat((labels, class_labels), dim=1)
        self.n_class = 0

        if self.icsd_label:
            icsd = torch.from_numpy(data['in_icsd'].values)
            labels = torch.concat((labels, icsd.unsqueeze(-1)), dim=1)
        
        if self.semic_label:
            semic = torch.from_numpy(data['band_gap'].values)
            semic = torch.where(semic<=0, 0, 1)
            labels = torch.concat((labels, semic.unsqueeze(-1)), dim=1)

        if self.class_label or self.icsd_label or self.semic_label:
            labels = labels[:, 1:]
        return labels, cond_bin
    
    def get_features(self, comp, max_num=8):
        feature = torch.zeros((max_num+1, len(self.elements_list)))
        feature[0, :] = 1
        for key, value in comp.as_dict().items():
            feature[int(value), self.elements_list.index(key)] = 1
            feature[0, self.elements_list.index(key)] = 0
        return feature

    @staticmethod
    def get_elements_list(data):
        elements = list()
        for i in data['composition']:
            list(i.as_dict().keys())
            elements += list(i.as_dict().keys())
            elements = list(set(elements))
        return elements
    
    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        x = self.features.iloc[index].unsqueeze(dim=0)
     
        label = self.labels[index, :]
        return x, label
        
