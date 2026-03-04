import torch
from torch.utils.data import Dataset
import h5py

class baselineDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        with h5py.File(self.h5_path, "r") as f:
                
            # --- load sample ---
            sample = f[self.keys[index]]

            # --- load and cast raw features from sample ---
            csm = torch.from_numpy(sample["csm"][:]).squeeze().to(torch.complex64) # (N, N), complex64
            coords = torch.from_numpy(sample["cartesian_coordinates"][:]).T.to(torch.float32) # (N, 3), float32 
            loc = torch.from_numpy(sample["loc"][:]).to(torch.float32) # (3, nsources), float32
            source_strength = torch.from_numpy(sample["source_strength_analytic"][:]).squeeze(0).to(torch.float32) # (nsources,), float32

            # --- normalize raw features ---
            dists_to_center = torch.norm(coords[:, :2], dim=1) # find microphones closest to center for normalization
            ref_idx = torch.argmin(dists_to_center)
            csm = csm / csm[ref_idx, ref_idx].real

            source_strength = source_strength / source_strength.sum()

            # --- compute eigenmodes ---
            eigvals, eigvecs = torch.linalg.eigh(csm) # (N,), (N, N)
            eigmode = eigvals[None,:] * eigvecs # (N, N)
            eigmode = torch.cat((eigmode.real, eigmode.imag), dim=-1) # (N, 2N), float32
            
            # --- labels ---
            loc_strongest_source = loc[:,torch.argmax(source_strength)]
            loc_strongest_source = loc_strongest_source[:2] # [2,]
            
            strength_strongest_source = source_strength[torch.argmax(source_strength)].unsqueeze(0) # [1,] 

            return eigmode, loc_strongest_source, strength_strongest_source


