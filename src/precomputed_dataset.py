import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data 
from torch_geometric.utils import dense_to_sparse

import h5py

class precomputedDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())

        self._edge_cache = {} # cache for fully connected edges to improve performance
        #self._f = None  # open file once 

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        # --- open file ---
        #f = self._get_file()

        with h5py.File(self.h5_path, "r") as f:
                
            # --- load sample ---
            sample = f[self.keys[index]]

            # --- load and cast raw features from sample ---
            csm = torch.from_numpy(sample["csm"][:]).squeeze().to(torch.complex64) # (N, N), complex64
            eigmode = torch.from_numpy(sample["eigmode"][:]).to(torch.complex64) # (N, N), complex64
            eigmode = torch.view_as_real(eigmode).to(torch.float32)  
            coords = torch.from_numpy(sample["cartesian_coordinates"][:]).T.to(torch.float32) # (N, 3), float32 
            loc = torch.from_numpy(sample["loc"][:]).to(torch.float32) # (3, nsources), float32
            source_strength = torch.from_numpy(sample["source_strength_analytic"][:]).squeeze(0).to(torch.float32) # (nsources,), float32


            # --- normalize raw features ---
            #TODO: check alternative approach normalize autopower by trace and cross spectra by coherence
            csm = csm / torch.trace(csm).real
            source_strength = source_strength / source_strength.sum()

            # --- define node features ---        
            theta = torch.atan2(coords[:, 1], coords[:, 0])
            cos_theta = torch.cos(theta) # (N,), float32
            sin_theta = torch.sin(theta) # (N,), float32

            r = torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2) # (N,), float32
            r = r / (r.max() + 1e-8) # normalize radius  
            
            autopower = torch.diagonal(csm) # (N,), complex64
            autopower_real = autopower.real # (N,), float32
            autopower_imag = autopower.imag # (N,), float32

            #TODO: implement positional encoding (Min-Sang Baek, Joon-Hyuk Chang, and Israel Cohen) 
    
            # --- define adjacency--- 
            N = coords.size(0)
            edge_index = self.get_fully_connected_edges(N)   # (2, E), cached, no self-loops

            src, dst = edge_index  # (E,), (E,)

            # --- define edge features ---
            cross_spectra = csm[src, dst]  # (E, 1), complex64
            cross_spectra_real = cross_spectra.real # (E, 1), float32
            cross_spectra_imag = cross_spectra.imag # (E, 1), float32

            dx = (coords[dst, 0] - coords[src, 0])
            dy = (coords[dst, 1] - coords[src, 1])   
            dist = torch.sqrt(dx**2 + dy**2 + 1e-8) # (E, 1), float32
            
            unit_direction_x = dx / dist # (E, 1), float32 
            unit_direction_y = dy / dist # (E, 1), float32

            cos_sim = (cos_theta[src] * cos_theta[dst] + sin_theta[src] * sin_theta[dst]) # (E, 1), float32, computed with trigonometric identity

            #TODO: implement directional features (Jingjie Fan, Rongzhi Gu, Yi Luo, and Cong Pang)


            # --- build feature vectors ---
            node_feat = self.build_feature(coords, r, cos_theta, sin_theta, autopower_real, autopower_imag, dim=1) # (N, F_node)
            edge_attr = self.build_feature(cross_spectra_real,cross_spectra_imag, dist, unit_direction_x, unit_direction_y, cos_sim, dim=1)  # (E, F_edge)

            # ---  define eigmode tokens analog to Kujawaski et. al---
            eigmode = torch.cat([torch.cat([eigmode[..., 0], -eigmode[..., 1]], dim=-1), torch.cat([eigmode[..., 1],  eigmode[..., 0]], dim=-1),],dim=-2,)

            # --- labels ---
            loc_strongest_source = loc[:,torch.argmax(source_strength)]
            loc_strongest_source = loc_strongest_source[:2].unsqueeze(0).unsqueeze(0) # [B,1,2]
            
            strength_strongest_source = source_strength[torch.argmax(source_strength)] 

            # --- build PyG Data ---
            data = Data(
                x=node_feat,                 # (N, F_node)
                edge_index=edge_index,       # (2, E)
                edge_attr=edge_attr,         # (E, F_edge)
                #TODO: Change to multiple sources and strengths later on
                y=loc_strongest_source,      # label used by training loop
            )

            data.eigmode = eigmode

            return data
    

    #--- utility functions ---
    @staticmethod
    def build_feature(*feats, dim=-1):
        """
        Utility function to construct a feature tensor from multiple inputs.

        If a tensor is 1D (shape: [N]), it is automatically expanded to
        shape [N, 1] so that it can be concatenated with higher-dimensional
        feature tensors.

        Parameters
        ----------
        *feats : torch.Tensor
            Feature tensors to be combined. Must be broadcast-compatible
            except for the concatenation dimension.
        dim : int, optional
            Dimension along which to concatenate the features (default: -1).

        Returns
        -------
        torch.Tensor
            Concatenated feature tensor.
        """
        feats = [feature.unsqueeze(-1) if feature.dim() == 1 else feature for feature in feats]
        return torch.cat(feats, dim=dim)

    def _get_file(self):
        """
        Lazily opens the HDF5 file and keeps it open for reuse
        to avoids repeatedly opening and closing the HDF5 file on every
        __getitem__ call. Reduces I/O overhead.

        """
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def get_fully_connected_edges(self, N):
        """
        Returns the edge_index of a fully connected directed graph with N nodes,
        excluding self-loops and caches the result for performance.

        Parameters
        ----------
        N : int
            Number of nodes in the graph.

        Returns
        -------
        edge_index : torch.Tensor
            Edge index tensor 
        """
        if N not in self._edge_cache:
            adj = torch.ones(N, N, dtype=torch.bool)
            adj.fill_diagonal_(False)
            self._edge_cache[N] = dense_to_sparse(adj)[0]

        return self._edge_cache[N]
