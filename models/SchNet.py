import torch
from torch_geometric.nn.models import SchNet

class tmqm_SchNet(SchNet):
    def forward(self,z,pos,batch,pool='raw'):
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            
        if pool == 'raw':
            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            if self.dipole:
                # Get center of mass.
                mass = self.atomic_mass[z].view(-1, 1)
                M = self.sum_aggr(mass, batch, dim=0)
                c = self.sum_aggr(mass * pos, batch, dim=0) / M
                h = h * (pos - c.index_select(0, batch))

            if not self.dipole and self.mean is not None and self.std is not None:
                h = h * self.std + self.mean

            if not self.dipole and self.atomref is not None:
                h = h + self.atomref(z)

            out = self.readout(h, batch, dim=0)

            if self.dipole:
                out = torch.norm(out, dim=-1, keepdim=True)

            if self.scale is not None:
                out = self.scale * out

            return out
        else:
            return h