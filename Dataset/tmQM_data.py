import numpy as np
import torch
import copy
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
import pandas as pd
import pickle

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')  

energy_level = ('3d','4s','4p','4d','4f','5s','5p','5d','5f','5g','6s','6p','6d','6f','6g','6h','7s')

def order_round(order):
    if order < 0.5:
        return 0
    elif order <= 1.25:
        return 1
    elif order <= 1.75:
        return 1.5
    elif order <= 2.25:
        return 2
    elif order <= 2.75:
        return 2.5
    elif order <= 3.25:
        return 3
    else:
        print('No define order!')

def distance(xyz):
    xyz = np.array(xyz)
    dist = []
    for i in range(xyz.shape[0]):
        dist.append(np.linalg.norm(xyz[i,:]-xyz,axis=1))
    return np.array(dist)

def DG(xyz,dim):
    x = copy.deepcopy(xyz)
    dist = distance(x)
    center = np.sum(x,axis=0)/x.shape[0]
    
    x = x-center
    do = np.linalg.norm(x,axis=1)
    do2 = np.tile(do**2,len(do)).reshape(-1,len(do))
    G = (do2+do2.T-dist**2)/2
    val,vec = np.linalg.eig(G)
    val = np.real(val)
    vec = np.real(vec)
    ind = np.argsort(val)[::-1][:dim]
    eigv = np.array([np.sqrt(abs(i)) if i!=0 else 0 for i in val[ind]])
    coor = vec[:,ind] * eigv

    if coor.shape[1] < dim:
        coor = np.pad(coor,((0,0),(0,dim-coor.shape[1])),mode='constant')

    return coor

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

def pos_encode(coor=None,sigma=1,dim=3,type = 'laplace'):
    if type == 'laplace':
        w = copy.deepcopy(coor)
        non_zero = np.nonzero(coor)
        w[non_zero] = np.exp(-w[non_zero]**2/(2*sigma**2))
        d = np.diag(np.sum(w,axis=1))
        L = d-w
        val,vec = np.linalg.eig(L)
        sel = []
        val_sort = np.argsort(val)
        for i in val_sort:
            if val[i] > 1e-8:
                sel.append(i)
            if len(sel) == dim:
                break
        pe = vec[:,sel]
        if len(sel)<dim:
            pe = np.pad(pe,((0,0),(0,dim-len(sel))),mode='constant')
    elif type == 'DG':
        pe = DG(coor,dim)
    else:
        print('Undefined position code!')

    return pe

class tmQM_dataset(Dataset):
    def __init__(self,path,heter=False,pe='laplace',metal_list=None,index_list=None,round=False,label_index=None):        
        super().__init__()
        with open(path['complex'],'rb') as f:
            temp = pickle.load(f)
        if metal_list != None:
            metal_list = [periodic_table[idx] for idx in metal_list]
            self.complexes = []
            metal_idx = []
            for idx,comp in enumerate(temp):
                if comp['metal'] in metal_list:
                    metal_idx.append(idx)
                    self.complexes.append(comp)
        else:
            self.complexes = temp
        
        self.metal_level = pd.read_csv(path['metal_level'],delimiter=',',
                                        usecols=['AtomicNumber','Symbol','ElectronConfiguration']).to_numpy()
        with open(path['entirety'],'rb') as f:
            temp = pickle.load(f)
        if metal_list != None:
            self.adjs = [temp[idx] for idx in metal_idx]
        else:
            self.adjs = temp

        self.pe = pe
        self.round = round
        self.heter = heter
        self.label_index = label_index
        self.data_list = self.process(index_list)

    def process_step(self,index):
        if self.heter:
            atoms = self.complexes[index]['atoms']
            adj = self.adjs[index]
            atomic_number = [periodic_table.index(i)+1 for i in atoms]

            del atomic_number[self.complexes[index]['metal_pos']]
            x1 = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)
            metal = query_level(self.complexes[index]['metal'],self.metal_level)
            metal.append(self.metal_level[np.squeeze(
                np.where(self.metal_level[:,1] == self.complexes[index]['metal'])),0])
            metal = torch.tensor([metal])

            heter_graph = HeteroData()

            metal_adj = np.delete(adj[self.complexes[index]['metal_pos'],:],self.complexes[index]['metal_pos'])[:,np.newaxis]
            ligand_adj = np.delete(adj,self.complexes[index]['metal_pos'],0)
            ligand_adj = np.delete(ligand_adj,self.complexes[index]['metal_pos'],1)
            l2m = np.argwhere(metal_adj!=0)
            l2l = np.argwhere(ligand_adj!=0)

            heter_graph['ligand','bond','ligand'].edge_index = torch.tensor(l2l.T,dtype=torch.int64)
            heter_graph['ligand','donor','metal'].edge_index = torch.tensor(l2m.T,dtype=torch.int64)
            heter_graph['metal','backbonding','ligand'].edge_index = torch.tensor(l2m[:,[1,0]].T,dtype=torch.int64)

            l2l_attr = []
            l2m_attr = []
            m2l_attr = []

            if self.pe != 'off':
                if self.pe == 'laplace':
                    dist = self.complexes[index]['dist']
                    sub_xyz = pos_encode(dist,type=self.pe)
                    sub_xyz = sub_xyz.tolist()
                elif self.pe == 'DG':
                    xyz = self.complexes[index]['xyz']
                    sub_xyz = np.array(xyz)
                    sub_xyz = pos_encode(sub_xyz,type=self.pe)
                    sub_xyz = sub_xyz.tolist()
                else:
                    sub_xyz = self.complexes[index]['xyz']
                
                metal_xyz = [sub_xyz[self.complexes[index]['metal_pos']]]
                del sub_xyz[self.complexes[index]['metal_pos']]
                ligand_xyz = sub_xyz
                x2 = torch.tensor(ligand_xyz,dtype=torch.float32)
                heter_graph['ligand'].x = torch.cat([x1,x2],dim=-1)

                for pos in l2l:
                    edge_xyz = [ligand_xyz[pos[1]][i] - ligand_xyz[pos[0]][i] for i in range(3)]
                    if self.round:
                        edge_xyz.append(order_round(ligand_adj[pos[0],pos[1]]))
                    else:
                        edge_xyz.append(ligand_adj[pos[0],pos[1]])
                    l2l_attr.append(edge_xyz)
                
                for pos in l2m:
                    edge_xyz = [metal_xyz[pos[1]][i] - ligand_xyz[pos[0]][i] for i in range(3)]
                    if self.round:
                        # 配位键无简化键级
                        pass
                    else:
                        edge_xyz.append(metal_adj[pos[0],pos[1]])
                    l2m_attr.append(edge_xyz)

                for pos in l2m[:,[1,0]]:
                    edge_xyz = [ligand_xyz[pos[1]][i] - metal_xyz[pos[0]][i] for i in range(3)]
                    if self.round:
                        # 配位键无简化键级
                        pass
                    else:
                        edge_xyz.append(metal_adj[pos[1],pos[0]])
                    m2l_attr.append(edge_xyz)
            else:
                heter_graph['metal'].x = metal
                heter_graph['ligand'].x = x1
                for pos in l2l:
                    if self.round:
                        l2l_attr.append(order_round(ligand_adj[pos[0],pos[1]]))
                    else:
                        l2l_attr.append(ligand_adj[pos[0],pos[1]])

                for pos in l2m:
                    if self.round:
                        # 配位键无简化键级
                        pass
                    else:
                        l2m_attr.append(metal_adj[pos[0],pos[1]])

                for pos in l2m[:,[1,0]]:
                    if self.round:
                        # 配位键无简化键级
                        pass
                    else:
                        m2l_attr.append(metal_adj[pos[0],pos[1]])

            y = torch.tensor(self.complexes[index]['y'],dtype=torch.float32)
            l2l_attr = torch.tensor(l2l_attr,dtype=torch.float32)
            l2m_attr = torch.tensor(l2m_attr,dtype=torch.float32)
            m2l_attr = torch.tensor(m2l_attr,dtype=torch.float32)

            metal_xyz = torch.tensor(metal_xyz,dtype=torch.float32)
            heter_graph['metal'].x =  torch.cat([metal,metal_xyz],dim=-1)

            heter_graph['ligand','bond','ligand'].edge_attr = l2l_attr
            heter_graph['ligand','donor','metal'].edge_attr = l2m_attr
            heter_graph['metal','backbonding','ligand'].edge_attr = m2l_attr

            heter_graph.pos = torch.tensor(self.complexes[index]['xyz'],dtype=torch.float32)
            if self.label_index == -1:
                heter_graph.y = y
            else:
                heter_graph.y = y[self.label_index].unsqueeze(dim=0).unsqueeze(dim=0)
            heter_graph.sample_ind = index

            return heter_graph
        else:
            atoms = self.complexes[index]['atoms']
            adj = self.adjs[index]
            atomic_number = [periodic_table.index(i)+1 for i in atoms]

            x1 = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)
            edge_attr = []
            bond_pos = np.argwhere(adj!=0)
            edge_index = torch.tensor(bond_pos.T,dtype=torch.int64)

            if self.pe != 'off':
                if self.pe == 'laplace':
                    dist = self.complexes[index]['dist']
                    sub_xyz = pos_encode(dist,type=self.pe)
                elif self.pe == 'DG':
                    xyz = self.complexes[index]['xyz']
                    sub_xyz = np.array(xyz)
                    sub_xyz = pos_encode(sub_xyz,type=self.pe)
                else:
                    sub_xyz = self.complexes[index]['xyz']
                    
                x2 = torch.tensor(sub_xyz,dtype=torch.float32).view(-1,3)
                x = torch.cat([x1,x2],dim=-1)

                for pos in bond_pos:
                    edge_xyz = [sub_xyz[pos[1]][i] - sub_xyz[pos[0]][i] for i in range(3)]
                    if self.round:
                        edge_xyz.append(order_round(adj[pos[0],pos[1]]))
                    else:
                        edge_xyz.append(adj[pos[0],pos[1]])
                    edge_attr.append(edge_xyz)
            else:
                x = x1
                for pos in bond_pos:
                    if self.round:
                        edge_attr.append([order_round(adj[pos[0],pos[1]])])
                    else:
                        edge_attr.append([adj[pos[0],pos[1]]])

            y = torch.tensor(self.complexes[index]['y'],dtype=torch.float32)
            edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
            graph = Data(x,edge_index,edge_attr)
            graph.pos = torch.tensor(self.complexes[index]['xyz'],dtype=torch.float32)

            metal = query_level(self.complexes[index]['metal'],self.metal_level)
            metal.append(self.metal_level[np.squeeze(
                np.where(self.metal_level[:,1] == self.complexes[index]['metal'])),0])
            metal = torch.tensor([metal],dtype=torch.float32)
            graph.metal = metal

            if self.label_index == -1:
                graph.y = y
            else:
                graph.y = y[self.label_index].unsqueeze(dim=0).unsqueeze(dim=0)
            graph.sample_ind = index

            return graph
        
    def process(self,index_list):
        data_list = []
        if index_list != None:
            for index in index_list:
                data_list.append(self.process_step(index))
        else:
            sample_num = len(self.complexes)
            for index in range(sample_num):
                data_list.append(self.process_step(index))

        return data_list
            
    def __getitem__(self,index):
        return self.data_list[index]
        
    def __len__(self):
        return len(self.data_list)
                
class tmQM_wrapper(object):
    def __init__(self,path,batch_size=32,num_workers=0,valid_size=0.1,test_size=0.1,pe='laplace',predata=None,ORound=False,heter=False,label_index=None):
        super(object,self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.path = path
        self.predata = predata
        self.pe = pe
        self.heter = heter
        self.round = ORound
        self.label_index = label_index

    def get_data_loaders(self):
        if self.predata != None:
            dataset = self.predata
        else:
            dataset = tmQM_dataset(self.path,self.heter,self.pe,round=self.round,label_index=self.label_index)
            
        num_data = len(dataset)
        indices = list(range(num_data))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_data))
        split2 = int(np.floor(self.test_size * num_data))
        valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]

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