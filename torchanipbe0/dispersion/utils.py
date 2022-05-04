# The neural network dispersion utilities
# Author: Tu Nguyen Thien Phuc
# Last update: 2021-09-30

import torch
import numpy as np
from typing import NamedTuple
from torch import Tensor

BOHR_TO_ANSTROM = 0.529177249

class CAtomic(NamedTuple):
    atoms: Tensor
    coefficients: Tensor

def cutoff_function(distance, cutoff):
    '''
    Return the cutoff function tensor from distance function
    Only work with constant cutoff
    ro is 0.66 of rc by default
    '''
    rc = cutoff
    ro = 0.66 * cutoff
    if not isinstance(distance, torch.Tensor):
        # Error check
        raise TypeError('Distance should be in torch Tensor')
    assert distance.dim() == 2
    r = distance **2
    rc = rc**2
    ro = ro**2
    return torch.where(
                 r < ro,
                 torch.ones(r.size(), device = r.device, dtype = r.dtype),
                 torch.where( r > rc ,
                              torch.zeros(r.size(), device = r.device, dtype = r.dtype),
                              (rc - r)**2 * (rc + 2*r - 3*ro) / (rc-ro) **3))


def cell_info(cell, pbc, cut_off):
    '''
    Return the cell information
    '''
    cell_inv = torch.linalg.pinv(cell)
    surface_dist_inv = cell_inv.norm(2,0)
    n_bins = torch.ceil(cut_off * surface_dist_inv).to(torch.long)
    n_bins = torch.where(pbc, n_bins, n_bins.new_zeros(n_bins.shape))
    return cell_inv, surface_dist_inv, n_bins

def compute_shifts(cell, pbc, cut_off):
    '''
    Return the shift (in interger) possible for the computational
    Mimic the original compute_shifts from TorchANI
    '''
    _, _, n_bins = cell_info(cell, pbc, cut_off)
    r1 = torch.arange(1, n_bins[0].item() + 1, device = cell.device)
    r2 = torch.arange(1, n_bins[1].item() + 1, device = cell.device)
    r3 = torch.arange(1, n_bins[2].item() + 1, device = cell.device)
    o = torch.zeros(1, dtype = torch.long, device = cell.device)
    # Return the expansion from any direction
    return torch.cat([
        torch.cartesian_prod(r1, r2, r3),
        torch.cartesian_prod(r1, r2, o),
        torch.cartesian_prod(r1, r2, -r3),
        torch.cartesian_prod(r1, o, r3),
        torch.cartesian_prod(r1, o, o),
        torch.cartesian_prod(r1, o, -r3),
        torch.cartesian_prod(r1, -r2, r3),
        torch.cartesian_prod(r1, -r2, o),
        torch.cartesian_prod(r1, -r2, -r3),

        torch.cartesian_prod(o, r2, r3),
        torch.cartesian_prod(o, r2, o),
        torch.cartesian_prod(o, r2, -r3),
        torch.cartesian_prod(o, o, r3),
        # torch.cartesian_prod(o, o, o), # Not self-reflect
        torch.cartesian_prod(o, o, -r3),
        torch.cartesian_prod(o, -r2, r3),
        torch.cartesian_prod(o, -r2, o),
        torch.cartesian_prod(o, -r2, -r3),

        torch.cartesian_prod(-r1, r2, r3),
        torch.cartesian_prod(-r1, r2, o),
        torch.cartesian_prod(-r1, r2, -r3),
        torch.cartesian_prod(-r1, o, r3),
        torch.cartesian_prod(-r1, o, o),
        torch.cartesian_prod(-r1, o, -r3),
        torch.cartesian_prod(-r1, -r2, r3),
        torch.cartesian_prod(-r1, -r2, o),
        torch.cartesian_prod(-r1, -r2, -r3)
    ])

def neighbor_list(coor, cell, pbc, cut_off):
    n_batch, n_atoms, _ = coor.shape
    # Compute self interaction
    index_inside = torch.triu_indices(n_atoms, n_atoms, 1, dtype = torch.long, device = cell.device)
    shift_inside = torch.zeros((index_inside.shape[1],3),dtype = torch.long, device = cell.device)
    # Compute the interaction with super cell
    index_temp = torch.triu_indices(n_atoms, n_atoms, device = cell.device)
    shift_temp = compute_shifts(cell, pbc, cut_off)
    n_ind = index_temp.shape[1]
    n_shift = shift_temp.shape[0]
    index_outside = index_temp.repeat(1,n_shift)
    shift_outside = shift_temp.unsqueeze(1).expand(-1,n_ind,-1).reshape((-1,3))
    # Combine the inside and outside
    index_all = torch.cat([index_inside, index_outside], dim = 1)
    shift_all = torch.cat([shift_inside, shift_outside])
    shift = torch.matmul(shift_all.to(torch.float), cell)
    coordinate = coor.index_select(1, index_all.view(-1)).view(n_batch, 2, -1, 3)
    dist = (coordinate[:,1,:,:] - coordinate[:,0,:,:] + shift.view(1,-1,3)).norm(2, -1) # Implicit broadcast here for n_batch
    # Take the interaction if any batch have that in the cut_off distance
    ok_label = (dist <= cut_off).any(dim = 0)
    return dist[:, ok_label], index_all[:, ok_label]
