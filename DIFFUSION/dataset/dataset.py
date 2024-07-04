import torch
from torch.utils.data import Dataset
import pickle
class D2_Dataset(Dataset):

    def __init__(self, root='./data/2D.pickle', data_shape=(8, 84)) -> None:
        super().__init__()
        self.root = root
        with open(root, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        x = torch.Tensor(self.data.iloc[index, 0]).unsqueeze(dim=0)
        lable = torch.Tensor(self.data.iloc[index, 1])
        return x, lable
    