from torch.utils.data import Dataset
from deepfake_utils import compute_fft_spectrum

class FFTDataset(Dataset):
    def __init__(self,base_dataset):
        self.base=base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self,idx):
        img_tensor,label=self.base[idx]
        fft_map=compute_fft_spectrum(
            img_tensor
        )
        return fft_map,label