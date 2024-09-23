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
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from tqdm import tqdm

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')  

energy_level = ('3d','4s','4p','4d','4f','5s','5p','5d','5f','5g','6s','6p','6d','6f','6g','6h','7s')

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

class MOF_dataset(Dataset):
    def __init__(self,path,database,label_index=None,pe='DG',data_list=None,index_list=None):        
        super().__init__()
        prop = pd.read_csv(path['root']+database+'_prop.csv',header=None)
        self.metal_level = pd.read_csv(path['metal_level'],delimiter=',',
                                usecols=['AtomicNumber','Symbol','ElectronConfiguration']).to_numpy()
        self.y = prop.iloc[:,label_index+1].to_list()
        self.cif_name = prop.iloc[:,0]
        self.cif_path = path[database]
        self.label_index = label_index
        self.pe = pe
        self.index_list = index_list
        if data_list == None:
            self.data_list = self.process()
        else:
            self.data_list = data_list
            self.reset()

    def process_step(self,index):
        crystal = Structure.from_file(self.cif_path+self.cif_name[index]+'.cif')
        max_num_nbr=8
        radius=8
        atomic_number = [crystal[i].specie.number for i in range(len(crystal))]
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                # warnings.warn('{} not find enough neighbors to build graph. '
                #                 'If it happens frequently, consider increase '
                #                 'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                    [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                [radius + 1.] * (max_num_nbr -
                                                        len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))
        nbr_fea_idx, nbr_fea = nbr_fea_idx, nbr_fea
        nbr_fea = np.array(nbr_fea)
        nbr_fea_idx = nbr_fea_idx
        
        x = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)

        start = []
        end = []
        for ind,nbrs in enumerate(nbr_fea_idx):
            for nbr in nbrs:
                start.append(ind)
                end.append(nbr)
        edge_index = torch.tensor([start,end])
        edge_attr = torch.tensor(nbr_fea.reshape(-1,1),dtype=torch.float32)

        if self.pe == 'DG':
            coor = crystal.cart_coords
            sub_xyz = pos_encode(coor,type='DG')
        else:
            sub_xyz = coor
        edge_xyz = []
        for i in range(len(start)):
            edge_xyz.append([sub_xyz[start[i]][j] - sub_xyz[end[i]][j] for j in range(3)])
        edge_attr = torch.cat([torch.tensor(edge_xyz,dtype=torch.float32),edge_attr],dim=1)

        graph = Data(x,edge_index,edge_attr)
        graph.pos = torch.tensor(coor,dtype=torch.float32)

        cry_metal = []
        for site in crystal:
            element = Element(site.specie.symbol)
            if element.is_metal:
                cry_metal.append(site.specie.symbol)
        if len(cry_metal) == 0:
            return -1
        cry_metal = cry_metal[0]

        if periodic_table.index(cry_metal)+1 < 21:
            return -1

        metal = query_level(cry_metal,self.metal_level)
        metal.append(self.metal_level[np.squeeze(
            np.where(self.metal_level[:,1] == cry_metal)),0])
        metal = torch.tensor([metal],dtype=torch.float32)
        graph.metal = metal

        graph.y = torch.tensor(self.y[index],dtype=torch.float32)
        graph.sample_ind = index

        return graph
        
    def process(self):
        data_list = []
        sample_num = len(self.y)
        for index in tqdm(range(sample_num)):
            graph = self.process_step(index)
            if graph == -1:
                pass
            else:
                data_list.append(graph)

        return data_list
    
    def get_subset(self,index_list):
        temp = []
        for ind in index_list:
            temp.append(self.data_list[ind])
        self.data_list = temp

    def reset(self):
        if type(self.y[0]) == str:
            for i in range(len(self.data_list)):
                self.data_list[i].y = torch.tensor(eval(self.y[self.data_list[i].sample_ind]),dtype=torch.float32)
        else:
            for i in range(len(self.data_list)):
                self.data_list[i].y = torch.tensor(self.y[self.data_list[i].sample_ind],dtype=torch.float32)

    def __getitem__(self,index):
        return self.data_list[index]
        
    def __len__(self):
        return len(self.data_list)
                
class MOF_wrapper(object):
    def __init__(self,path,batch_size=32,num_workers=0,valid_size=0.1,test_size=0.1,pe='laplace',database='QMOF',predata=None,label_index=None):
        super(object,self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.path = path
        self.predata = predata
        self.database = database
        self.pe = pe
        self.label_index = label_index

    def get_data_loaders(self):
        if self.predata != None:
            dataset = self.predata
        else:
            dataset = MOF_dataset(self.path,self.database,label_index=self.label_index,pe=self.pe)
            
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