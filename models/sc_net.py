import torch
import math
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear,Parameter,ModuleList

from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing,HeteroConv, HANConv,GATv2Conv,SAGEConv
from torch_geometric.utils import to_dense_batch,to_dense_adj,add_self_loops,degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool,DenseSAGEConv, dense_diff_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import HGTConv

from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GIN
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import PNA
from torch_geometric.nn.models import GraphSAGE
from models.SchNet import tmqm_SchNet
from models.AttentiveFP import tmqm_AttentiveFP
from torch_geometric.nn.models import DimeNet
from torch_geometric.nn.models import DimeNetPlusPlus


num_atom_type = 119

metadata = (['ligand', 'metal'],
            [('ligand', 'bond', 'ligand'),
            ('ligand', 'donor', 'metal'),
            ('metal', 'backbonding', 'ligand')])

def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class GNNConv(MessagePassing):
    def __init__(self,hidden_channels,edge_dim):
        super().__init__(aggr='add')

        self.mlp = Seq(
            Linear(2*hidden_channels,hidden_channels),
            nn.ReLU()
        )

        # self.lin = Linear(2*in_channels,in_channels)

        if edge_dim == 1:
            self.edge_weight1 = Linear(1,hidden_channels,bias=False)
        elif edge_dim == 4:
            self.edge_weight1 = Linear(1,hidden_channels//2,bias=False)
            self.edge_weight2 = Linear(3,hidden_channels//2,bias=False)

        self.mlp.apply(self.reset_parameters)
        self.edge_dim = edge_dim

    def reset_parameters(self,m):
        if type(m) == Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self,x,edge_index,edge_attr):
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # self_loop_attr = torch.zeros(x.size(0), 4)
        # self_loop_attr[:,3] = 4 # bond-order
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_index,_ = gcn_norm(edge_index)
        
        if self.edge_dim == 1:
            edge_fea = self.edge_weight1(edge_attr)
        elif self.edge_dim == 4:
            edge_fea = torch.cat([self.edge_weight1(edge_attr[:,3:]),self.edge_weight2(edge_attr[:,0:3])],dim=-1)
        elif self.edge_dim == 0:
            edge_fea = None

        edge_index,_ = gcn_norm(edge_index)
        row, col = edge_index[0],edge_index[1]
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, edge_attr=edge_fea,norm=norm)
        out = self.mlp(torch.cat([x,out],dim=1))

        return out

    def message(self, x_j: Tensor,edge_attr,norm) -> Tensor:
        if self.edge_dim == 0:
            return norm.view(-1,1)*(x_j)
        else:
            return norm.view(-1,1)*(x_j + edge_attr)
        
class MPNN(nn.Module):
    def __init__(self,num_layer=5,hidden_channel=256,edge_dim=4,drop_ratio=0) -> None:
        super(MPNN,self).__init__()
        self.num_layer = num_layer
        self.hidden_channel = hidden_channel
        self.drop_ratio = drop_ratio

        # nn.init.xavier_uniform_(self.x_embedding.weight.data)
        # nn.init.xavier_uniform_(self.x_weight.data)
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GNNConv(hidden_channel,edge_dim))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channel))
        
    def forward(self,x,edge_index,edge_attr):
        h=x
        for layer in range(self.num_layer):
            h = self.gnns[layer](h,edge_index,edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        return h

class diff_pool(nn.Module):
    def __init__(self,d_feature=256,class_num=16) -> None:
        super().__init__()

        self.SAGE = DenseSAGEConv(d_feature,abs(class_num))
        self.pool_ln = nn.BatchNorm1d(abs(class_num))

    def forward(self,h,batch,edge_index):
        dense_h,mask = to_dense_batch(h,batch)
        adj = to_dense_adj(edge_index,batch)
        s = self.SAGE(dense_h,adj,mask).permute(0,2,1)
        s = self.pool_ln(s).relu().permute(0,2,1)
        x, _, _, _ = dense_diff_pool(dense_h, adj, s, mask)

        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = torch.softmax(s, dim=-1)

        return x,adj,s,mask
    
class metal_attention(nn.Module):
    def __init__(self,d_feature=256,d_k=256,kernel='exp',interprate=False,drop_ratio=0) -> None:
        super().__init__()

        self.W_q = nn.Linear(d_feature,d_k,bias=False)
        self.W_k = nn.Linear(d_feature,d_k,bias=False)
        self.W_v = nn.Linear(d_feature,d_k,bias=False)
        self.lynorm = nn.LayerNorm(d_k)
        self.dk = d_k
        self.d_feature = d_feature
        self.kernel = kernel
        self.interprate = interprate
        self.drop_ratio = drop_ratio

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self,metal_feature,x,mask=None):
        Q = self.W_q(metal_feature)
        K = self.W_k(x)
        V = self.W_v(x)
        if mask != None:
            mask = mask.int().unsqueeze(-2)
            attention = torch.bmm(torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk)-(1-mask)*1e10),V)
        else:
            if self.kernel == 'exp':
                attention = torch.bmm(torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk)),V)
            elif self.kernel == 'RBF':
                attention = torch.bmm(torch.nn.Softmax(dim=-1)(-torch.norm(torch.sub(Q,K),p=2,dim=-1).unsqueeze(1)/math.sqrt(self.dk)),V)

        if self.interprate:
            weight = torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk))
            return attention.relu().squeeze(1),weight
        else:
            return F.dropout(attention.relu().squeeze(1), self.drop_ratio, training=self.training)
    
class CoordiPool(nn.Module):
    def __init__(self,hidden_channel,d_k,apool=16,drop_ratio=0,kernel='exp',interprate=False) -> None:
        super().__init__()
        if apool != -1:
            self.apool_layer = diff_pool(hidden_channel,apool)
        else:
            pass
        self.attention = metal_attention(hidden_channel,d_k,kernel,interprate)
        self.kernel = kernel
        self.drop_ratio = drop_ratio
        self.apool = apool
        self.interprate = interprate
        if apool == -16:
            self.ligand_layer = nn.Linear(hidden_channel,d_k)

    def forward(self,x,metal_feature,batch,edge_index):
        metal_feature = metal_feature.unsqueeze(dim=-2)
        if self.apool != -1:
            h,_,s,_= self.apool_layer(x,batch,edge_index)
        else:
            h,mask = to_dense_batch(x,batch)

        if self.interprate and self.apool > 0:
            h,weight = self.attention(metal_feature,h)
            return h,weight,s
        else:
            if self.apool == -1:
                h = self.attention(metal_feature,h,mask=mask)
            elif self.apool == -16:
                h = torch.mean(h,dim=1)
                h = self.ligand_layer(h)
            else:
                h = self.attention(metal_feature,h)
            return h

class MLHGT(nn.Module):
    def __init__(self,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels,
        class_num=None,
        dropout: float = 0.0,
        heads=2,
        readout='cat',
        pool='mean',
        conv='hgt') -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.readout=readout
        self.conv = conv

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels
        
        if conv == 'hgt':
            self.convs = ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HGTConv(in_channels,hidden_channels,metadata,heads))
        elif conv == 'MPNN':
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('ligand', 'bond', 'ligand'): SAGEConv(-1,hidden_channels),
                    ('ligand', 'donor', 'metal'): SAGEConv((-1,-1),hidden_channels,add_self_loops=False),
                    ('metal', 'backbonding', 'ligand'): SAGEConv((-1,-1),hidden_channels,add_self_loops=False)
                }, aggr='sum')
                self.convs.append(conv)
        elif conv == 'han':
            self.convs = ModuleList()
            for _ in range(num_layers):
                self.convs.append(
                    HANConv(in_channels,hidden_channels,metadata,heads))
            
        self.x_embedding = nn.Embedding(num_atom_type,int(hidden_channels/2))
        self.pos_weight = Linear(3,int(hidden_channels/2),bias=False)
        self.metal_weight = Linear(17,int(hidden_channels/2),bias=False)

        nn.init.xavier_uniform_(self.x_embedding.weight.data)

        if self.readout == 'cat':
            self.pre_head = Seq(
                Linear(2*hidden_channels,hidden_channels),
                nn.Softplus(),
                Linear(hidden_channels,out_channels)
            )
        elif self.readout == 'add' or self.readout == 'ligand':
            self.pre_head = Seq(
                Linear(hidden_channels,hidden_channels),
                nn.Softplus(),
                Linear(hidden_channels,out_channels)
            )
        elif self.readout == 'attention':
            self.pool = diff_pool(hidden_channels,class_num=class_num)
            self.metal_attention = metal_attention(hidden_channels,)
            self.pre_head = Seq(
                Linear(hidden_channels,hidden_channels),
                nn.Softplus(),
                Linear(hidden_channels,out_channels)
            )

        if self.readout == 'attention':
            pass
        else:
            if pool == 'mean':
                self.pool = global_mean_pool
            elif pool == 'add':
                self.pool = global_add_pool
            elif pool == 'max':
                self.pool = global_max_pool
            else:
                raise ValueError('Not defined pooling!')
        
    def forward(self,x_dict,edge_index_dict,batch_dict):
        h_dict = {}
        h_dict['ligand'] = torch.cat([self.x_embedding(x_dict['ligand'][:,0].int()),self.pos_weight(x_dict['ligand'][:,1:])],dim=1)
        h_dict['metal'] = torch.cat([self.metal_weight(x_dict['metal'][:,:17])
                                     +self.x_embedding(x_dict['metal'][:,17].int()),
                                     self.pos_weight(x_dict['metal'][:,18:])],dim=1)
        if self.conv == 'hgt':
            for hgt in self.convs:
                h_dict = hgt(h_dict,edge_index_dict)
        elif self.conv == 'MPNN':
            for hgt in self.convs:
                h_dict = hgt(h_dict,edge_index_dict)
        
        if self.readout == 'cat':
            h_dict['ligand'] = self.pool(h_dict['ligand'],batch_dict['ligand'])
            h_dict['metal'] = self.pool(h_dict['metal'],batch_dict['metal'])
            z = torch.cat([h_dict['ligand'],h_dict['metal']],dim=-1)
            out = self.pre_head(z)
        elif self.readout == 'add':
            h_dict['ligand'] = self.pool(h_dict['ligand'],batch_dict['ligand'])
            h_dict['metal'] = self.pool(h_dict['metal'],batch_dict['metal'])
            z = h_dict['ligand']+h_dict['metal']
            out = self.pre_head(z)
        elif self.readout == 'attention':
            h_dict['ligand'],_,_,_ = self.pool(h_dict['ligand'],batch_dict['ligand'],edge_index_dict[('ligand', 'bond', 'ligand')])
            h_dict['metal'] = h_dict['metal'].unsqueeze(dim=1)
            z = self.metal_attention(h_dict['metal'],h_dict['ligand'])
            out = self.pre_head(z)
        elif self.readout == 'ligand':
            z = self.pool(h_dict['ligand'],batch_dict['ligand'])
            out = self.pre_head(z)

        return out


class TMCnet(nn.Module):
    def __init__(self,GNN_config,Heter_config,out_dimention,gnn='GCN',deg=None) -> None:
        super(TMCnet,self).__init__()
        if gnn == 'GCN':
            self.GNN = GCN(
                in_channels=GNN_config['hidden_channels'],
                hidden_channels=GNN_config['hidden_channels'],
                out_channels=None,
                num_layers=GNN_config['num_layer'],
                dropout=GNN_config['drop_ratio'],
                act='relu',
                norm="BatchNorm"
            )
        elif gnn == 'GIN':
            self.GNN = GIN(in_channels=GNN_config['hidden_channels'],
                           hidden_channels=GNN_config['hidden_channels'],
                           out_channels=None,
                           num_layers=GNN_config['num_layer'],
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm")
        elif gnn == 'GAT':
            self.GNN = GAT(in_channels=GNN_config['hidden_channels'],
                           hidden_channels=GNN_config['hidden_channels'],
                           edge_dim=GNN_config['edge_dim'],
                           out_channels=None,
                           num_layers=GNN_config['num_layer'],
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm")
        elif gnn == 'GAT-v2':
            self.GNN = GAT(in_channels=GNN_config['hidden_channels'],
                           hidden_channels=GNN_config['hidden_channels'],
                           edge_dim=GNN_config['edge_dim'],
                           out_channels=None,
                           num_layers=GNN_config['num_layer'],
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm",
                           v2=True)
        elif gnn == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            self.GNN = PNA(in_channels=GNN_config['hidden_channels'],
                           hidden_channels=GNN_config['hidden_channels'],
                           edge_dim=GNN_config['edge_dim'],
                           out_channels=None,
                           num_layers=GNN_config['num_layer'],
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm",
                           aggregators=aggregators,
                           scalers=scalers,
                           deg=deg)
        elif gnn == 'GraphSAGE':
            self.GNN = GraphSAGE(
                in_channels=GNN_config['hidden_channels'],
                hidden_channels=GNN_config['hidden_channels'],
                out_channels=None,
                num_layers=GNN_config['num_layer'],
                dropout=GNN_config['drop_ratio'],
                act='relu',
                norm="BatchNorm",
            )
        elif gnn == 'MPNN':
            self.GNN = MPNN(
                num_layer=GNN_config['num_layer'],
                hidden_channel=GNN_config['hidden_channels'],
                edge_dim=GNN_config['edge_dim'],
                drop_ratio=GNN_config['drop_ratio']
            )
        elif gnn == 'SchNet':
            self.GNN = tmqm_SchNet(hidden_channels=GNN_config['hidden_channels'],
                                   num_filters=GNN_config['hidden_channels'])
            self.GNN.lin2 = nn.Linear(self.GNN.hidden_channels // 2, out_dimention)
        elif gnn == 'AttentiveFP':
            if GNN_config['pool'] == 'attention':
                out_channel = GNN_config['hidden_channels']
            else:
                out_channel = out_dimention
            self.GNN = tmqm_AttentiveFP(
                in_channels=GNN_config['hidden_channels'],
                hidden_channels=GNN_config['hidden_channels'],
                out_channels=out_channel,     
                num_layers=GNN_config['num_layer'],
                dropout=GNN_config['drop_ratio'],
                edge_dim=GNN_config['edge_dim'],
                num_timesteps=2   
            )
        elif gnn == 'DimeNet':
            self.GNN = DimeNet(
                hidden_channels=GNN_config['hidden_channels'],
                out_channels=out_dimention,
                num_blocks=3,
                num_bilinear=3,
                num_spherical=3,
                num_radial=3
            )
        elif gnn == 'DimeNet++':
            self.GNN = DimeNetPlusPlus(
                hidden_channels=GNN_config['hidden_channels'],
                out_channels=out_dimention,
                num_blocks=3,
                int_emb_size=32,
                basis_emb_size=4,
                out_emb_channels=GNN_config['hidden_channels'],
                num_spherical=3,
                num_radial=3
            )
        elif gnn == 'MLHGT':
            self.GNN = MLHGT(
                in_channels=GNN_config['hidden_channels'],
                hidden_channels=GNN_config['hidden_channels'],
                num_layers=GNN_config['num_layer'],
                out_channels=out_dimention,
                conv=Heter_config['conv'],
                readout=GNN_config['pool'],
                class_num=GNN_config['apool']
            )

        self.GNN_config = GNN_config
        self.gnn = gnn
        self.interprate = GNN_config['interprate']
        
        self.pre_head = Seq(
            Linear(GNN_config['hidden_channels'],2*GNN_config['hidden_channels']),
            nn.Softplus(),
            Linear(2*GNN_config['hidden_channels'],out_dimention)
        )

        if GNN_config['in_channels'] == 1:
            self.x_embedding = nn.Embedding(num_atom_type,GNN_config['hidden_channels'])
        else:
            self.x_embedding = nn.Embedding(num_atom_type,int(GNN_config['hidden_channels']/2))
            self.pos_weight = Linear(3,int(GNN_config['hidden_channels']/2),bias=False)
        
        if GNN_config['pool'] == 'attention':
            self.metal_embedding = nn.Embedding(num_atom_type,int(GNN_config['hidden_channels']/2))
            self.metal_weight = Linear(17,int(GNN_config['hidden_channels']/2),bias=False)
            self.metal_layer = Linear(GNN_config['hidden_channels'],GNN_config['hidden_channels'],bias=False)

        nn.init.xavier_uniform_(self.x_embedding.weight.data)

        if GNN_config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif GNN_config['pool'] == 'add':
            self.pool = global_add_pool
        elif GNN_config['pool'] == 'max':
            self.pool = global_max_pool
        elif GNN_config['pool'] == 'attention':
            self.CoordiPool = CoordiPool(GNN_config['hidden_channels'],
                                         GNN_config['hidden_channels'],
                                         GNN_config['apool'],
                                         GNN_config['drop_ratio'],
                                         interprate=GNN_config['interprate'])
        elif GNN_config['pool'] == 'raw' or GNN_config['pool'] == 'cat':
            pass
        else:
            raise ValueError('Not defined pooling!')

    def forward(self,data):
        if self.gnn != 'MLHGT':
            if self.GNN_config['in_channels']==1:
                h = self.x_embedding(data.x[:,0].int())
            elif self.GNN_config['in_channels']==4:
                h = torch.cat([self.x_embedding(data.x[:,0].int()),self.pos_weight(data.x[:,1:])],dim=1)
            batch = data.batch
            edge_index = data.edge_index
            if self.GNN_config['edge_dim'] == 1:
                if data.edge_attr.shape[1]>1:
                    edge_attr = data.edge_attr[:,3:]
                else:
                    edge_attr = data.edge_attr
            elif self.GNN_config['edge_dim'] == 4:
                edge_attr = data.edge_attr
            elif self.GNN_config['edge_dim'] == 0:
                edge_attr = None

            if self.gnn == 'SchNet' or (self.gnn == 'DimeNet' or self.gnn == 'DimeNet++'):
                h = self.GNN(data.x[:,0].int(),data.pos,data.batch,pool=self.GNN_config['pool'])
                if self.GNN_config['pool'] == 'raw':
                    out = h
                elif self.GNN_config['pool'] == 'attention':
                    metal_feature = torch.cat([self.metal_embedding(data['metal'][:,17].int()),
                                               self.metal_weight(data['metal'][:,:17])],dim=1)
                    metal = self.metal_layer(metal_feature)
                    if self.GNN_config['mp'] == 1:
                        out = self.CoordiPool(h,metal_feature,batch,edge_index) + metal   
                    else:
                        out = self.CoordiPool(h,metal_feature,batch,edge_index)
                    out = self.pre_head(out)
                else:
                    out = self.pool(h,batch)
                    out = self.pre_head(out)
            elif self.gnn == 'AttentiveFP':
                h = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr,batch=batch,pool=self.GNN_config['pool'])
                if self.GNN_config['pool'] == 'raw':
                    out = h
                elif self.GNN_config['pool'] == 'attention':
                    metal_feature = torch.cat([self.metal_embedding(data['metal'][:,17].int()),
                                               self.metal_weight(data['metal'][:,:17])],dim=1)
                    metal = self.metal_layer(metal_feature)
                    out = h + metal
                    out = self.pre_head(out)
            else:
                h = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr)
                if self.GNN_config['pool'] != 'attention':
                    out = self.pool(h,batch)
                else:
                    metal_feature = torch.cat([self.metal_embedding(data['metal'][:,17].int()),
                                               self.metal_weight(data['metal'][:,:17])],dim=1)
                    metal = self.metal_layer(metal_feature)
                    if self.GNN_config['mp'] == 1:
                        if self.interprate:
                            h,weight,s = self.CoordiPool(h,metal_feature,batch,edge_index) 
                        else:
                            h = self.CoordiPool(h,metal_feature,batch,edge_index)
                        out = h + metal
                    else:
                        out = self.CoordiPool(h,metal_feature,batch,edge_index)
                out = self.pre_head(out)
                if self.interprate:
                    return out,weight,s
        else:
            out = self.GNN(x_dict=data.x_dict,
                           edge_index_dict=data.edge_index_dict,
                           batch_dict=data.batch_dict)

        return out