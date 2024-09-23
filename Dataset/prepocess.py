import pandas as pandas
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem
import pandas as pd
import pickle

# Read the information of bonds from tmQM_X.BO

# BO_path = './Data/tmQM_X.BO'
# X_path = './Data/tmQM_X.xyz'
# Y_path = './Data/tmQM_y.csv'

tran_metal = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
              'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yd','Lu',
              'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg']

def distance(xyz):
    xyz = np.array(xyz)
    dist = []
    for i in range(xyz.shape[0]):
        dist.append(np.linalg.norm(xyz[i,:]-xyz,axis=1))
    return np.array(dist)

def read_complex(BO_path,X_path,Y_path):
    '''read .BO flies'''
    complexes = []
    with open(BO_path) as f:
        lines = f.readlines()
        for l in tqdm(range(len(lines))):
            lsplit = lines[l].strip().split()
            if len(lsplit) == 0:
                continue
            if lsplit[0] == 'CSD_code':
                complexes.append({'CSD_code':lsplit[2],'metal':None,'metal_pos':-1,'atoms':[],'bonds':[[],[]],'bond_order':[]})
            else:
                start = int(lsplit[0])-1
                for ind,item in enumerate(lsplit):
                    if ind == 1:
                        complexes[-1]['atoms'].append(item)
                        if item in tran_metal:
                            complexes[-1]['metal_pos']=start
                            complexes[-1]['metal']=item

                    if ind > 2:
                        if (ind-4) % 3 == 0:
                            complexes[-1]['bonds'][0].append(start)
                            complexes[-1]['bonds'][1].append(int(item)-1)
                        elif (ind-5) % 3 == 0:
                            if float(item) > 0.5 or lsplit[1] in tran_metal:
                                # coordinate bonds and bonds with high bond order(>0.5)
                                # will be regraded as a bond in graph  
                                complexes[-1]['bond_order'].append(float(item))
                            else:
                                complexes[-1]['bonds'][0].pop()
                                complexes[-1]['bonds'][1].pop()
                    else:
                        pass

    with open(X_path) as f:
        lines = f.readlines()
        index = -1
        for l in tqdm(range(len(lines))):
            lsplit = lines[l].strip().split()
            if len(lsplit) == 0 or lsplit[0] == 'CSD_code':
                continue
            if len(lsplit) == 1:
                index += 1
                complexes[index]['xyz'] = []
                continue
            xyz = (float(lsplit[1]),float(lsplit[2]),float(lsplit[3]))
            complexes[index]['xyz'].append(xyz)

    y = pd.read_csv(Y_path,delimiter=';').to_numpy()
    for ind in tqdm(range(len(complexes))):
        complexes[ind]['y'] = y[ind,1:].astype(float)
        # Electronic_E
        # Dispersion_E
        # Dipole_M
        # Metal_q
        # HL_Gap
        # HOMO_Energy
        # LUMO_Energy
        # Polarizability
        complexes[ind]['dist'] = distance(complexes[ind]['xyz'])

    metal_deletion = []
    for ind in range(len(complexes)):
        if complexes[ind]['metal_pos'] == -1:
            metal_deletion.append(ind)
    return complexes,metal_deletion

def is_isomorphic(adjs,adj2):
    if adjs != []:
        for adj in adjs:
            g1 = nx.Graph(adj)
            g2 = nx.Graph(adj2)
            if nx.is_isomorphic(g1,g2):
                return True
    return False

def get_entirety(complexes):
    entirety = []
    for complex in tqdm(complexes):
        node_num = len(complex['atoms'])
        adjoin = np.zeros((node_num,node_num))
        bonds = complex['bonds']
        bonds_order = complex['bond_order']
        for i in range(len(bonds[0])):
            adjoin[bonds[0][i],bonds[1][i]] = bonds_order[i]
        for ind,bond in enumerate(adjoin[0,:]):
            adjoin[ind,0] = bond
        entirety.append(adjoin)
    return entirety     

def get_subgraph(complexes):
    subadj_set = []
    subgraph_set = []
    start_set = []
    for complex in tqdm(complexes):
        node_num = len(complex['atoms'])
        adjoin = np.zeros((node_num,node_num))
        bonds = complex['bonds']
        bonds_order = complex['bond_order']
        metal_pos = complex['metal_pos']
        for i in range(len(bonds[0])):
            adjoin[bonds[0][i],bonds[1][i]] = bonds_order[i]
        for ind,bond in enumerate(adjoin[0,:]):
            adjoin[ind,0] = bond

        coordinate = adjoin[metal_pos,:]
        adjoin = np.delete(adjoin,metal_pos,0)
        adjoin = np.delete(adjoin,metal_pos,1)

        start = np.where(coordinate!=0)[0]
        temp = []
        for s in start:
            if s <= metal_pos:
                temp.append(s)
            else:
                temp.append(s-1)
        start = temp
        seen = []
        subgraphs = []
        subadjs = []
        for s in start:
            if s in seen:
                continue
            stack = []
            subgraph = []
            stack.append(s)
            seen.append(s)
            subgraph.append(s)
            while stack:
                n = stack.pop()
                neiber = np.where(adjoin[n,:]!=0)[0]
                for n in neiber:
                    if n not in seen:
                        stack.append(n)
                        seen.append(n)
                        subgraph.append(n)
            subadj = adjoin[subgraph,:]
            subadj = subadj[:,subgraph]
            if subgraph != []:
                if is_isomorphic(subadjs,subadj):
                    continue
                else:
                    subadjs.append(subadj)
                    subgraphs.append(subgraph)
        subadj_set.append(subadjs)
        subgraph_set.append(subgraphs)
        start_set.append(start)
    return subgraph_set,subadj_set,start_set


def draw_ligands(index,complexes,adjs,nodes,sub_ind=None):
    print(complexes[index]['CSD_code'])
    atoms = [complexes[index]['atoms'][i] 
             for i in range(len(complexes[index]['atoms'])) 
             if i != complexes[index]['metal_pos']]
    adj = adjs[index]
    node = nodes[index]
    if sub_ind == None:
        fig,axes = plt.subplots(1,len(node),figsize=(16,8))
        node_code = 0
        for ind,ax in enumerate(axes):
            labels = {}
            for i,v in enumerate(node[ind]):
                node_code += 1
                labels[i] = atoms[v]+str(node_code)
            G = nx.Graph(adj[ind])
            pos = nx.spring_layout(G, seed=3113794652)
            nx.draw(G,pos,ax=ax,font_weight='bold')
            nx.draw_networkx_labels(G,pos,labels,ax=ax)
        plt.show()
    else:
        labels = {}
        for i,v in enumerate(node[sub_ind]):
            labels[i] = atoms[v]
        G = nx.Graph(adj[sub_ind])
        pos = nx.spring_layout(G, seed=3113794652)
        nx.draw(G,pos,font_weight='bold')
        nx.draw_networkx_labels(G,pos,labels)
        plt.show()

def draw_entirety(index,complexes,entirety):
    print(complexes[index]['CSD_code'])
    atoms = complexes[index]['atoms'] 
    adj = entirety[index]
    labels = {}
    for i,v in enumerate(atoms):
        labels[i] = v
    G = nx.Graph(adj)
    pos = nx.spring_layout(G, seed=3113794652)
    nx.draw(G,pos,font_weight='bold')
    nx.draw_networkx_labels(G,pos,labels)
    plt.show()


# complexes,metal_deletion = read_complex(BO_path,X_path,Y_path)
# nodes,adjs,start = get_subgraph(complexes)
# entirety = get_entirety(complexes)

# with open('./Data/complex.pkl','wb') as f:
#     pickle.dump(complexes,f)
# with open('./Data/nodes.pkl','wb') as f:
#     pickle.dump(nodes,f)
# with open('./Data/adjs.pkl','wb') as f:
#     pickle.dump(adjs,f)
# with open('./Data/entirety.pkl','wb') as f:
#     pickle.dump(entirety,f)
# with open('./Data/start.pkl','wb') as f:
#     pickle.dump(start,f)

# with open('./Data/complex.pkl','rb') as f:
#     complexes = pickle.load(f)
# with open('./Data/nodes.pkl','rb') as f:
#     nodes = pickle.load(f)
# with open('./Data/adjs.pkl','rb') as f:
#     adjs = pickle.load(f)
# with open('./Data/entirety.pkl','rb') as f:
#     entirety = pickle.load(f)
