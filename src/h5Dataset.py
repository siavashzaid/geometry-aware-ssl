import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data 
from torch_geometric.utils import dense_to_sparse

import h5py

class h5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f.keys())

        self._edge_cache = {} # cache for fully connected edges to improve performance
        self._f = None  # open file once 

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        # --- open file ---
        f = self._get_file()

        # --- load sample ---
        sample = f[self.keys[index]]

        # --- load raw features from sample ---
        csm = torch.from_numpy(sample["csm"][:]).squeeze() # (N, N), complex128
        eigmode = torch.from_numpy(sample["eigmode"][:]) # (N, N), complex128
        loc = torch.from_numpy(sample["loc"][:]) # (Dimensions, Num_Sources), float64
        source_strength = torch.from_numpy(sample["source_strength_analytic"][:]).squeeze(0) # (Num_Sources,), float64
        
        # --- define node features ---
        coords = torch.from_numpy(sample["cartesian_coordinates"][:]).T # (N, Dimensions), float64 
        
        theta = torch.atan2(coords[:, 1], coords[:, 0])
        cos_theta = torch.cos(theta) # (N,), float64
        sin_theta = torch.sin(theta) # (N,), float64

        r = torch.sqrt(coords[:, 0]**2 + coords[:, 1]**2) # (N,), float64
        r = r / (r.max() + 1e-8) # normalize radius  
        
        autopower = torch.diagonal(csm) # (N,), complex128
        autopower_real = autopower.real
        autopower_imag = autopower.imag

        #TODO: implement positional encoding

        # --- define adjacency--- #
        node_index = torch.arange(coords.shape[0])
        ii, jj = torch.meshgrid(node_index, node_index, indexing="ij") 
        mask = (ii != jj) #remove self-loops
        src = ii[mask].reshape(-1) 
        dst = jj[mask].reshape(-1)  

        edge_index = torch.stack([src, dst], dim=0) # (2, E)

        # --- define edge features ---
        cross_spectra = csm[mask]  # (E, 1), complex128
        cross_spectra_real = cross_spectra.real # (E, 1), float64
        cross_spectra_imag = cross_spectra.imag # (E, 1), float64

        dx = (coords[dst, 0] - coords[src, 0]).unsqueeze(-1)
        dy = (coords[dst, 1] - coords[src, 1]).unsqueeze(-1)     
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8) # (E, 1), float64
        
        unit_direction_x = dx / dist # (E, 1), float64 
        unit_direction_y = dy / dist # (E, 1), float64

        cos_sim = (cos_theta[src] * cos_theta[dst] + sin_theta[src] * sin_theta[dst]) # (E, 1), float64, computed with trigonometric identity

        #TODO: implement positional encoding


        # --- build feature vectors ---

        node_feat = self.build_feature(coords, r, cos_theta, sin_theta, autopower_real, autopower_imag, dim=1) # (N, F_node)
        edge_attr = self.build_feature(cross_spectra_real,cross_spectra_imag, dist, unit_direction_x, unit_direction_y, cos_sim, dim=1)  # (E, F_edge)

        # --- labels ---
        loc_strongest_source = loc[:,torch.argmax(source_strength)]


        # --- build PyG Data ---
        data = Data(
            x=node_feat,                 # (N, F_node)
            edge_index=edge_index,       # (2, E)
            edge_attr=edge_attr,         # (E, F_edge)
            y=loc_strongest_source,      # label used by training loop
        )

        data.eigmode = eigmode

        return data#, eigmode
    

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
