import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd

import ast
import unicodedata
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')  

energy_level = ('3d','4s','4p','4d','4f','5s','5p','5d','5f','5g','6s','6p','6d','6f','6g','6h','7s')

METAL_PATH = './Dataset/PubChemElements_all.csv'

def query_level(metal,metal_level):
    level_dic = []
    level = metal_level[np.where(metal_level[:,1]==metal)[0],2][0]
    for item in level.split(' '):
        if item[0] == '[' and len(item) == 4:
            continue
        elif item[0] == '[' and len(item) > 4:
            level_dic.append({
                'level':item[4:6],
                'ele_num':int(item[6])
            })
        else:
            level_dic.append({
                'level':item[0:2],
                'ele_num':int(item[2])
            })
    level_emb = [0 for _ in range(len(energy_level))]
    for item in level_dic:
        level_emb[energy_level.index(item['level'])] = item['ele_num']
    return level_emb

def combine_graph(graph_list):
    combined_graph = Data()
    combined_graph.x = torch.cat([data.x for data in graph_list], dim=0)
    combined_edge_indices = []
    combined_edge_attrs = []
    node_split = []
    node_offset = 0
    sub_num = len(graph_list)
    for ind,data in enumerate(graph_list):
        num_nodes = data.num_nodes
        node_split.append((sub_num-ind-1)*torch.ones((num_nodes,1),dtype=torch.int))
        # Adjust edge indices based on node offset
        edge_index = data.edge_index.clone()
        edge_index[0] += node_offset
        edge_index[1] += node_offset
        combined_edge_indices.append(edge_index)

        # Concatenate edge attributes
        combined_edge_attrs.append(data.edge_attr)
        node_offset += num_nodes
    combined_graph.edge_index = torch.cat(combined_edge_indices, dim=1)
    combined_graph.edge_attr = torch.cat(combined_edge_attrs, dim=0)
    combined_graph.node_split = torch.cat(node_split, dim=0)
    combined_graph.subgraph_num = torch.tensor(len(graph_list))

    return combined_graph

def simul_order(type):
    if type == BT.SINGLE:
        return 1.0
    elif type == BT.DOUBLE:
        return 2.0
    elif type == BT.TRIPLE:
        return 3.0
    elif type == BT.AROMATIC:
        return 1.5

def read_data():
    with open('./Dataset/MOFs.pkl','rb') as f:
        data = pickle.load(f)

    return data

class hMOF_dataset(Dataset):
    def __init__(self,type='atom',cat='cat0'):
        self.metal_level = pd.read_csv(METAL_PATH,delimiter=',',
                                           usecols=['AtomicNumber','Symbol','ElectronConfiguration']).to_numpy()
        RDLogger.DisableLog('rdApp.*')

        self.MOFs = read_data()
        self.type = type
        self.cat = cat
 
        self.data_list = self.process()

    def process_step(self,index):
        graphs = []

        y = torch.tensor(self.MOFs[index]['adsorption'],dtype=torch.float32)
        metal = query_level(self.MOFs[index]['metal'],self.metal_level)
        metal.append(self.metal_level[np.squeeze(
        np.where(self.metal_level[:,1] == self.MOFs[index]['metal'])),0])
        metal = torch.tensor(metal)
        
        for ind,smi in enumerate(self.MOFs[index]['linker']):
            try:
                mol = Chem.MolFromSmiles(smi)
                atom = mol.GetAtoms()
            except:
                mol = Chem.MolFromSmarts(smi)
            try:
                mol = Chem.AddHs(mol)
            except:
                pass
            atomic_number = []
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())

            x = torch.tensor(atomic_number).view(-1,1)

            row, col, edge_feat = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]

                edge_xyz1 = [simul_order(bond.GetBondType())]
                edge_xyz2 = [simul_order(bond.GetBondType())]
                edge_feat.append(edge_xyz1)
                edge_feat.append(edge_xyz2)
                
            edge_index = torch.tensor([row, col], dtype=torch.int64)
            edge_attr = torch.tensor(np.array(edge_feat),dtype=torch.float32)

            data = Data(x,edge_index,edge_attr)
            graphs.append(data)

        if len(graphs) == 0:
            print(F"ERROR SAMPLE:{self.MOFs[index]['linker']}")
            return -1
        else:
            combined_graph = combine_graph(graphs)
            combined_graph.metal = metal
            # if len(y.shape) != 2:
            #     y = y.unsqueeze(dim=1)
            combined_graph.y = y
            combined_graph.name = self.MOFs[index]['name']

        return combined_graph
        
    def process(self):
        print('Dataset generating......')
        data_list = []
        error_list = []
        samples = len(self.MOFs)
        for index in range(samples):
            if (self.MOFs[index]['type'] != self.type and self.type != 'both') or self.MOFs[index]['cat'] != self.cat:
                continue
            else:
                data = self.process_step(index)
                if data == -1:
                    error_list.append(index)
                else:
                    data_list.append(data)

        return data_list

    def __getitem__(self,index):
        return self.data_list[index]
    
    def __len__(self):
        return len(self.data_list)

class hMOF_wrapper(object):
    def __init__(self,batch_size=32,num_workers=0,valid_size=0.1,test_size=0.1,predata=None,split='random',type='atom',cat='cat0'):
        super(object,self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.split = split
        self.predata = predata
        self.type = type
        self.cat = cat

    def get_data_loaders(self):
        dataset = self.predata
            
        num_data = len(dataset)
        indices = list(range(num_data))

        if self.split == 'random':
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_data))
            split2 = int(np.floor(self.test_size * num_data))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_sampler = SubsetRandomSampler(test_idx)

        elif self.split == 'uniform':
            y = dataset.data['y']
            index = np.argsort(y)
            interpration = int(1/self.test_size)
            train_idx = []
            valid_idx = []
            test_idx = []
            for i,value in enumerate(index):
                if (i+1) % interpration == 0 and (i+1) != len(index):
                    test_idx.append(value)
                else:
                    train_idx.append(value)
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader,valid_loader,test_loader
    
if __name__ == '__main__':
    dataset = hMOF_dataset()