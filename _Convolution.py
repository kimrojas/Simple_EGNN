import math

from e3nn import o3
from e3nn.o3 import TensorProduct, Linear, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet
from e3nn.util.jit import compile_mode

import torch
from torch_scatter import scatter

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """
    def __init__(
        self,
        irreps_in,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_out,
        number_of_basis,
        radial_layers,
        radial_neurons,
        num_neighbors=None
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        
        #First Linear layer: processing node feature V
        self.lin1 = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_in,
            internal_weights=True,
            shared_weights=True,
        )

        #Create instruction for tensor product
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
                        
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        #Tensor product between node feature V and edge attribute Y (angular part: spherical harmonic): weight . (V x Y) 
        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        #MLP for radial part of edge (the weight for above Tensor product)
        self.fc = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp
        
        self.lin2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
    
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedded) -> torch.Tensor:
        weight = self.fc(edge_length_embedded)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x)
        
        edge_features = self.tp(x[edge_src], edge_attr, weight)

        #avg_num_neigh: Optional[float] = self.num_neighbors
        #if avg_num_neigh is not None:
        #    edge_features = edge_features.div(avg_num_neigh**0.5)
        
        x = scatter(edge_features, edge_dst, dim=0, dim_size=x.shape[0]).div(self.num_neighbors**0.5)

        x = self.lin2(x)

        #c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        #m = self.sc.output_mask
        #c_x = (1 - m) + c_x * m
        #return c_s * s + c_x * x
        return x + s