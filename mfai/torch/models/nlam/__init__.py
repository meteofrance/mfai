"""
Graph Neural Network architectures adapted from https://github.com/mllam/neural-lam
"""

from ctypes import Union
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Generic, Literal, Tuple, TypeVar, overload

import torch
import torch_geometric as pyg
from dataclasses_json import dataclass_json
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper

from mfai.torch.models.base import BaseModel, ModelType
from mfai.torch.models.utils import expand_to_batch

from .create_mesh import build_graph_for_grid
from .interaction_net import InteractionNet, make_mlp


def offload_to_cpu(model: nn.ModuleList) -> nn.ModuleList:
    return nn.ModuleList([offload_wrapper(x) for x in model])


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no parameters and only
    buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors: list[torch.Tensor], persistent: bool=True) -> None:
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key: int) -> Any:
        return getattr(self, f"b{key}")

    def __len__(self) -> int:
        return self.n_buffers

    def __iter__(self) -> Generator[Any]:
        return (self[i] for i in range(len(self)))


def load_graph(graph_dir: Path, device: torch.device | Literal["cpu", "cuda"]="cpu") -> Tuple[bool, dict]:
    """
    Loads a graph from its disk serialised format into a dict of Tensors.
    """
    # Load edges (edge_index)
    m2m_edge_index: BufferList = BufferList(
        torch.load(graph_dir / "m2m_edge_index.pt", device), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index: torch.Tensor = torch.load(graph_dir / "g2m_edge_index.pt", device)  # (2, M_g2m)
    m2g_edge_index: torch.Tensor = torch.load(graph_dir / "m2g_edge_index.pt", device)  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    m2m_features: list[torch.Tensor] = torch.load(
        graph_dir / "m2m_features.pt", device
    )  # List of (M_m2m[l], d_edge_f)
    g2m_features: torch.Tensor = torch.load(
        graph_dir / "g2m_features.pt", device
    )  # (M_g2m, d_edge_f)
    m2g_features: torch.Tensor = torch.load(
        graph_dir / "m2g_features.pt", device
    )  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge: torch.Tensor = max(
        [torch.max(level_features[:, 0]) for level_features in m2m_features]
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features: list[torch.Tensor] = torch.load(
        graph_dir / "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert len(m2m_features) == n_levels, "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    mesh_up_edge_index: BufferList | list[torch.Tensor]
    mesh_down_edge_index : BufferList | list[torch.Tensor]
    mesh_up_features: BufferList | list[torch.Tensor]
    mesh_down_features: BufferList | list[torch.Tensor]
    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            torch.load(graph_dir / "mesh_up_edge_index.pt", device),
            persistent=False,
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            torch.load(graph_dir / "mesh_down_edge_index.pt", device),
            persistent=False,
        )  # List of (2, M_down[l])

        mesh_up_features = torch.load(
            graph_dir / "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = torch.load(
            graph_dir / "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [edge_features / longest_edge for edge_features in mesh_up_features],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [edge_features / longest_edge for edge_features in mesh_down_features],
            persistent=False,
        )

        mesh_static_features = BufferList(mesh_static_features, persistent=False)
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    print(f"Graph is hierarchical {hierarchical}")
    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }


@dataclass_json
@dataclass(slots=True)
class GraphLamSettings:
    """
    Settings for graph-based models
    """

    tmp_dir: Path = Path("/tmp")  # nosec B108
    hidden_dims: int = 64
    hidden_layers: int = 1

    use_checkpointing: bool = False
    offload_to_cpu: bool = False

    mesh_aggr: Literal["sum", "mean"] = "sum"
    processor_layers: int = 4

    def __str__(self) -> str:
        return f"ModelCOnfig : {self.hidden_dims}x{self.hidden_layers}x{self.processor_layers}"


class BaseGraphModel(BaseModel):
    """
    Base (abstract) class for graph-based models building on
    the encode-process-decode idea.
    """

    settings_kls = GraphLamSettings
    hierarchical = False
    onnx_supported = False
    supported_num_spatial_dims = (1,)
    num_spatial_dims: int = 1
    model_type = ModelType.GRAPH
    features_last: bool = True

    @classmethod
    def rank_zero_setup(cls, settings: GraphLamSettings, meshgrid: torch.Tensor) -> None:
        """
        This is a static method to allow multi-GPU
        trainig frameworks to call this method once
        on rank zero before instantiating the model.
        """
        # this doesn't take long and it prevents discrepencies
        build_graph_for_grid(
            meshgrid,
            settings.tmp_dir,
            hierarchical=cls.hierarchical,
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple,
        settings: GraphLamSettings,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_shape = input_shape
        self._settings = settings

        hierarchical, graph_ldict = load_graph(self.settings.tmp_dir)
        if hierarchical != self.hierarchical:
            raise ValueError(
                f"Loaded graph is {hierarchical} while expecting {self.hierarchical}"
            )
        for name, attr_value in graph_ldict.items():
            # Make BufferLists module members and register tensors as buffers
            if isinstance(attr_value, torch.Tensor):
                print(f"Registering buffer {name}")
                self.register_buffer(name, attr_value, persistent=False)
            else:
                print(f"setattr {name}")
                setattr(self, name, attr_value)


        self.g2m_edges: int = self.g2m_features.shape[0]
        g2m_dim: int = self.g2m_features.shape[1]
        self.m2g_edges: int = self.m2g_features.shape[0]
        m2g_dim: int = self.m2g_features.shape[1]

        # Define sub-models
        # Feature embedders for grid
        self.mlp_blueprint_end: list[int] = [self.settings.hidden_dims] * (
            self.settings.hidden_layers + 1
        )
        self.grid_embedder = make_mlp(
            [in_channels] + self.mlp_blueprint_end,
            checkpoint=self.settings.use_checkpointing,
        )
        self.g2m_embedder = make_mlp(
            [g2m_dim] + self.mlp_blueprint_end,
            checkpoint=self.settings.use_checkpointing,
        )
        self.m2g_embedder = make_mlp(
            [m2g_dim] + self.mlp_blueprint_end,
            checkpoint=self.settings.use_checkpointing,
        )

        # GNNs
        # encoder

        print(
            "Hideem_dims",
            self.settings.hidden_dims,
            "g2m_dim",
            [g2m_dim] + self.mlp_blueprint_end,
        )
        self.g2m_gnn = InteractionNet(
            self.g2m_edge_index,
            self.settings.hidden_dims,
            hidden_layers=self.settings.hidden_layers,
            update_edges=False,
            checkpoint=self.settings.use_checkpointing,
        )
        self.encoding_grid_mlp = make_mlp(
            [self.settings.hidden_dims] + self.mlp_blueprint_end,
            checkpoint=self.settings.use_checkpointing,
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            self.m2g_edge_index,
            self.settings.hidden_dims,
            hidden_layers=self.settings.hidden_layers,
            update_edges=False,
            checkpoint=self.settings.use_checkpointing,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = make_mlp(
            [self.settings.hidden_dims] * (self.settings.hidden_layers + 1)
            + [out_channels],
            layer_norm=False,
            checkpoint=self.settings.use_checkpointing,
        )  # No layer norm on this one

        # subclasses should override this method
        self.finalize_graph_model()
        self.check_required_attributes()

    @property
    def settings(self) -> GraphLamSettings:
        return self._settings

    def finalize_graph_model(self) -> None:
        """
        Method to be overridden by subclasses for finalizing the graph model
        """
        pass

    def get_num_mesh(self) -> tuple[int, int]:
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        raise NotImplementedError("get_num_mesh not implemented")

    def embedd_mesh_nodes(self) -> torch.Tensor:
        """
        Embedd static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        raise NotImplementedError("embedd_mesh_nodes not implemented")

    def process_step(self, mesh_rep: torch.Tensor) -> torch.Tensor:
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        raise NotImplementedError("process_step not implemented")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, N_grid, feature_dim), X_t
        prev_prev_state: (B, N_grid, feature_dim), X_{t-1}
        forcing: (B, N_grid, forcing_dim)
        """
        batch_size = x.shape[0]

        # print("Features",grid_features.dtype, grid_features.shape)
        # Embedd all features
        grid_emb = self.grid_embedder(x)  # (B, N_grid, d_h)
        g2m_emb = self.g2m_embedder(self.g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(self.m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes()

        # Map from grid to mesh
        mesh_emb_expanded = expand_to_batch(mesh_emb, batch_size)  # (B, N_mesh, d_h)
        g2m_emb_expanded = expand_to_batch(g2m_emb, batch_size)
        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb_expanded, g2m_emb_expanded
        )  # (B, N_mesh, d_h)
        # Also MLP with residual for grid representation
        grid_rep = grid_emb + self.encoding_grid_mlp(grid_emb)  # (B, N_grid, d_h)
        # Run processor step
        mesh_rep = self.process_step(mesh_rep)

        # Map back from mesh to grid
        m2g_emb_expanded = expand_to_batch(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb_expanded
        )  # (B, N_grid, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(grid_rep)  # (B, N_grid, d_f)
        return net_output


class BaseHiGraphModel(BaseGraphModel):
    """
    Base class for hierarchical graph models.
    """

    hierarchical = True

    def finalize_graph_model(self) -> None:
        # Track number of nodes, edges on each level
        # Flatten lists for efficient embedding
        self.N_levels = len(self.mesh_static_features)

        # Number of mesh nodes at each level
        self.N_mesh_levels = [
            mesh_feat.shape[0] for mesh_feat in self.mesh_static_features
        ]  # Needs as python list for later
        # N_mesh_levels_torch = torch.tensor(self.N_mesh_levels)

        # Print some useful info
        print("Loaded hierachical graph with structure:")
        for lvl, N_level in enumerate(self.N_mesh_levels):
            same_level_edges = self.m2m_features[lvl].shape[0]
            print(f"level {lvl} - {N_level} nodes, {same_level_edges} same-level edges")

            if lvl < (self.N_levels - 1):
                up_edges = self.mesh_up_features[lvl].shape[0]
                down_edges = self.mesh_down_features[lvl].shape[0]
                print(
                    f"  {lvl}<->{lvl+1} - {up_edges} up edges, {down_edges} down edges"
                )

        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = self.mesh_static_features[0].shape[1]
        mesh_same_dim = self.m2m_features[0].shape[1]
        mesh_up_dim = self.mesh_up_features[0].shape[1]
        mesh_down_dim = self.mesh_down_features[0].shape[1]

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [
                make_mlp(
                    [mesh_dim] + self.mlp_blueprint_end,
                    checkpoint=self.settings.use_checkpointing,
                )
                for _ in range(self.N_levels)
            ]
        )
        if self.settings.offload_to_cpu:
            self.mesh_embedders = offload_to_cpu(self.mesh_embedders)

        self.mesh_same_embedders = nn.ModuleList(
            [
                make_mlp(
                    [mesh_same_dim] + self.mlp_blueprint_end,
                    checkpoint=self.settings.use_checkpointing,
                )
                for _ in range(self.N_levels)
            ]
        )
        if self.settings.offload_to_cpu:
            self.mesh_same_embedders = offload_to_cpu(self.mesh_same_embedders)

        self.mesh_up_embedders = nn.ModuleList(
            [
                make_mlp(
                    [mesh_up_dim] + self.mlp_blueprint_end,
                    checkpoint=self.settings.use_checkpointing,
                )
                for _ in range(self.N_levels - 1)
            ]
        )
        if self.settings.use_checkpointing:
            self.mesh_up_embedders = offload_to_cpu(self.mesh_up_embedders)

        self.mesh_down_embedders = nn.ModuleList(
            [
                make_mlp(
                    [mesh_down_dim] + self.mlp_blueprint_end,
                    checkpoint=self.settings.use_checkpointing,
                )
                for _ in range(self.N_levels - 1)
            ]
        )
        if self.settings.offload_to_cpu:
            self.mesh_down_embedders = offload_to_cpu(self.mesh_down_embedders)

        # Instantiate GNNs
        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.settings.hidden_dims,
                    hidden_layers=self.settings.hidden_layers,
                    checkpoint=self.settings.use_checkpointing,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )
        if self.settings.use_checkpointing:
            self.mesh_init_gnns = offload_to_cpu(self.mesh_init_gnns)

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.settings.hidden_dims,
                    hidden_layers=self.settings.hidden_layers,
                    update_edges=False,
                    checkpoint=self.settings.use_checkpointing,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )
        if self.settings.offload_to_cpu:
            self.mesh_read_gnns = offload_to_cpu(self.mesh_read_gnns)

    def get_num_mesh(self) -> tuple[int, int]:
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        N_mesh = sum(node_feat.shape[0] for node_feat in self.mesh_static_features)
        N_mesh_ignore = N_mesh - self.mesh_static_features[0].shape[0]
        return N_mesh, N_mesh_ignore

    def embedd_mesh_nodes(self) -> torch.Tensor:
        """
        Embedd static mesh features
        This embedds only bottom level, rest is done at beginning of processing step
        Returns tensor of shape (N_mesh[0], d_h)
        """
        return self.mesh_embedders[0](self.mesh_static_features[0])

    def process_step(self, mesh_rep: torch.Tensor) -> torch.Tensor:
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        batch_size = mesh_rep.shape[0]

        # EMBEDD REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, N_mesh[l], d_h)
        mesh_rep_levels: list[torch.Tensor] = [mesh_rep] + [
            expand_to_batch(emb(node_static_features), batch_size)
            for emb, node_static_features in zip(
                list(self.mesh_embedders)[1:], list(self.mesh_static_features)[1:]
            )
        ]

        # - EMBEDD EDGES -
        # Embedd edges, expand with batch dimension
        mesh_same_rep = [
            expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(self.mesh_same_embedders, self.m2m_features)
        ]
        mesh_up_rep = [
            expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(self.mesh_up_embedders, self.mesh_up_features)
        ]
        mesh_down_rep = [
            expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(self.mesh_down_embedders, self.mesh_down_features)
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[level_l - 1]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(send_node_rep, rec_node_rep, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = new_node_rep  # (B, N_mesh[l], d_h)
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.N_levels - 2, -1, -1), reversed(self.mesh_read_gnns)
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[level_l + 1]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(send_node_rep, rec_node_rep, edge_rep)

            # Update node and edge vectors in lists
            mesh_rep_levels[level_l] = new_node_rep  # (B, N_mesh[l], d_h)

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, N_mesh[0], d_h)

    def hi_processor_step(
        self, mesh_rep_levels: list[torch.Tensor], mesh_same_rep: list[torch.Tensor], mesh_up_rep: list[torch.Tensor], mesh_down_rep: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        raise NotImplementedError("hi_process_step not implemented")


class GraphLAM(BaseGraphModel):
    """
    Full graph-based LAM model that can be used with different (non-hierarchical )graphs.
    Mainly based on GraphCast, but the model from Keisler (2022) almost identical.
    Used for GC-LAM and L1-LAM in Oskarsson et al. (2023).
    """

    register: bool = True

    def finalize_graph_model(self) -> None:
        if self.hierarchical:
            raise ValueError("GraphLAM does not use a hierarchical mesh graph")

        mesh_dim: int = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = make_mlp(
            [mesh_dim] + self.mlp_blueprint_end,
            checkpoint=self.settings.use_checkpointing,
        )
        self.m2m_embedder = make_mlp(
            [m2m_dim] + self.mlp_blueprint_end,
            checkpoint=self.settings.use_checkpointing,
        )

        # GNNs
        # processor
        processor_nets = [
            InteractionNet(
                self.m2m_edge_index,
                self.settings.hidden_dims,
                hidden_layers=self.settings.hidden_layers,
                aggr=self.settings.mesh_aggr,
                checkpoint=self.settings.use_checkpointing,
            )
            for _ in range(self.settings.processor_layers)
        ]
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [
                (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                for net in processor_nets
            ],
        )

    def get_num_mesh(self) -> tuple[int, int]:
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return len(self.mesh_static_features), 0

    def embedd_mesh_nodes(self) -> torch.Tensor:
        """
        Embedd static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)

    def process_step(self, mesh_rep: torch.Tensor) -> torch.Tensor:
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embedd m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(self.m2m_features)  # (M_mesh, d_h)
        m2m_emb_expanded = expand_to_batch(m2m_emb, batch_size)  # (B, M_mesh, d_h)

        mesh_rep, _ = self.processor(mesh_rep, m2m_emb_expanded)  # (B, N_mesh, d_h)
        return mesh_rep


class HiLAMParallel(BaseHiGraphModel):
    """
    Version of HiLAM where all message passing in the hierarchical mesh (up, down,
    inter-level) is ran in paralell.

    This is a somewhat simpler alternative to the sequential message passing of Hi-LAM.
    """

    register: bool = True

    def finalize_graph_model(self) -> None:
        super().finalize_graph_model()

        # Processor GNNs
        # Create the complete total edge_index combining all edges for processing
        total_edge_index_list = (
            list(self.m2m_edge_index)
            + list(self.mesh_up_edge_index)
            + list(self.mesh_down_edge_index)
        )
        total_edge_index = torch.cat(total_edge_index_list, dim=1)
        self.edge_split_sections = [ei.shape[1] for ei in total_edge_index_list]

        if self.settings.processor_layers == 0:
            self.processor = lambda x, edge_attr: (x, edge_attr)
        else:
            processor_nets = [
                InteractionNet(
                    total_edge_index,
                    self.settings.hidden_dims,
                    hidden_layers=self.settings.hidden_layers,
                    edge_chunk_sizes=self.edge_split_sections,
                    aggr_chunk_sizes=self.N_mesh_levels,
                )
                for _ in range(self.settings.processor_layers)
            ]
            self.processor = pyg.nn.Sequential(
                "mesh_rep, edge_rep",
                [
                    (net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")
                    for net in processor_nets
                ],
            )

    def hi_processor_step(
        self, mesh_rep_levels: list[torch.Tensor], mesh_same_rep: list[torch.Tensor], mesh_up_rep: list[torch.Tensor], mesh_down_rep: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """

        # First join all node and edge representations to single tensors
        mesh_rep = torch.cat(mesh_rep_levels, dim=1)  # (B, N_mesh, d_h)
        mesh_edge_rep = torch.cat(
            mesh_same_rep + mesh_up_rep + mesh_down_rep, axis=1
        )  # (B, M_mesh, d_h)

        # Here, update mesh_*_rep and mesh_rep
        mesh_rep, mesh_edge_rep = self.processor(mesh_rep, mesh_edge_rep)

        # Split up again for read-out step
        mesh_rep_levels = list(torch.split(mesh_rep, self.N_mesh_levels, dim=1))
        mesh_edge_rep_sections = torch.split(
            mesh_edge_rep, self.edge_split_sections, dim=1
        )

        mesh_same_rep = mesh_edge_rep_sections[: self.N_levels]
        mesh_up_rep = mesh_edge_rep_sections[
            self.N_levels : self.N_levels + (self.N_levels - 1)
        ]
        mesh_down_rep = mesh_edge_rep_sections[
            self.N_levels + (self.N_levels - 1) :
        ]  # Last are down edges

        # Note: We return all, even though only down edges really are used later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep


class HiLAM(BaseHiGraphModel):
    """
    Hierarchical graph model with message passing that goes sequentially down and up
    the hierarchy during processing.
    The Hi-LAM model from Oskarsson et al. (2023)
    """

    register: bool = True

    def finalize_graph_model(self) -> None:
        super().finalize_graph_model()

        # Make down GNNs, both for down edges and same level
        self.mesh_down_gnns = nn.ModuleList(
            [self.make_down_gnns() for _ in range(self.settings.processor_layers)]
        )  # Nested lists (proc_steps, N_levels-1)

        self.mesh_down_same_gnns = nn.ModuleList(
            [self.make_same_gnns() for _ in range(self.settings.processor_layers)]
        )  # Nested lists (proc_steps, N_levels)

        # Make up GNNs, both for up edges and same level
        self.mesh_up_gnns = nn.ModuleList(
            [self.make_up_gnns() for _ in range(self.settings.processor_layers)]
        )  # Nested lists (proc_steps, N_levels-1)

        self.mesh_up_same_gnns = nn.ModuleList(
            [self.make_same_gnns() for _ in range(self.settings.processor_layers)]
        )  # Nested lists (proc_steps, N_levels)

    def make_same_gnns(self) -> nn.ModuleList:
        """
        Make intra-level GNNs.
        """
        model = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.settings.hidden_dims,
                    hidden_layers=self.settings.hidden_layers,
                    checkpoint=self.settings.use_checkpointing,
                )
                for edge_index in self.m2m_edge_index
            ]
        )
        if self.settings.offload_to_cpu:
            model = offload_to_cpu(model)
        return model

    def make_up_gnns(self) -> nn.ModuleList:
        """
        Make GNNs for processing steps up through the hierarchy.
        """
        model = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.settings.hidden_dims,
                    hidden_layers=self.settings.hidden_layers,
                    checkpoint=self.settings.use_checkpointing,
                )
                for edge_index in self.mesh_up_edge_index
            ]
        )
        if self.settings.offload_to_cpu:
            model = offload_to_cpu(model)
        return model

    def make_down_gnns(self) -> nn.ModuleList:
        """
        Make GNNs for processing steps down through the hierarchy.
        """
        model = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    self.settings.hidden_dims,
                    hidden_layers=self.settings.hidden_layers,
                    checkpoint=self.settings.use_checkpointing,
                )
                for edge_index in self.mesh_down_edge_index
            ]
        )
        if self.settings.offload_to_cpu:
            model = offload_to_cpu(model)
        return model

    def mesh_down_step(
        self, mesh_rep_levels: list[torch.Tensor], mesh_same_rep: list[torch.Tensor], mesh_down_rep: list[torch.Tensor], down_gnns: nn.ModuleList, same_gnns: nn.ModuleList
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Run down-part of vertical processing, sequentially alternating between processing
        using down edges and same-level edges.
        """
        # Run same level processing on level L
        mesh_rep_levels[-1], mesh_same_rep[-1] = same_gnns[-1](
            mesh_rep_levels[-1], mesh_rep_levels[-1], mesh_same_rep[-1]
        )

        # Let level_l go from L-1 to 0
        for level_l, down_gnn, same_gnn in zip(
            range(self.N_levels - 2, -1, -1),
            reversed(down_gnns),
            reversed(same_gnns[:-1]),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[level_l + 1]  # (B, N_mesh[l+1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            down_edge_rep = mesh_down_rep[level_l]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply down GNN
            new_node_rep, mesh_down_rep[level_l] = down_gnn(
                send_node_rep, rec_node_rep, down_edge_rep
            )

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_down_rep

    def mesh_up_step(
        self, mesh_rep_levels: list[torch.Tensor], mesh_same_rep: list[torch.Tensor], mesh_up_rep: list[torch.Tensor], up_gnns: nn.ModuleList, same_gnns: nn.ModuleList
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Run up-part of vertical processing, sequentially alternating between processing
        using up edges and same-level edges.
        """

        # Run same level processing on level 0
        mesh_rep_levels[0], mesh_same_rep[0] = same_gnns[0](
            mesh_rep_levels[0], mesh_rep_levels[0], mesh_same_rep[0]
        )

        # Let level_l go from 1 to L
        for level_l, (up_gnn, same_gnn) in enumerate(
            zip(up_gnns, same_gnns[1:]), start=1
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[level_l - 1]  # (B, N_mesh[l-1], d_h)
            rec_node_rep = mesh_rep_levels[level_l]  # (B, N_mesh[l], d_h)
            up_edge_rep = mesh_up_rep[level_l - 1]
            same_edge_rep = mesh_same_rep[level_l]

            # Apply up GNN
            new_node_rep, mesh_up_rep[level_l - 1] = up_gnn(
                send_node_rep, rec_node_rep, up_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_up[l-1], d_h)

            # Run same level processing on level l
            mesh_rep_levels[level_l], mesh_same_rep[level_l] = same_gnn(
                new_node_rep, new_node_rep, same_edge_rep
            )
            # (B, N_mesh[l], d_h) and (B, M_same[l], d_h)

        return mesh_rep_levels, mesh_same_rep, mesh_up_rep

    def hi_processor_step(
        self, mesh_rep_levels: list[torch.Tensor], mesh_same_rep: list[torch.Tensor], mesh_up_rep: list[torch.Tensor], mesh_down_rep: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, N_mesh[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        for down_gnns, down_same_gnns, up_gnns, up_same_gnns in zip(
            self.mesh_down_gnns,
            self.mesh_down_same_gnns,
            self.mesh_up_gnns,
            self.mesh_up_same_gnns,
        ):
            # Down
            mesh_rep_levels, mesh_same_rep, mesh_down_rep = self.mesh_down_step(
                mesh_rep_levels, mesh_same_rep, mesh_down_rep, down_gnns, down_same_gnns
            )

            # Up
            mesh_rep_levels, mesh_same_rep, mesh_up_rep = self.mesh_up_step(
                mesh_rep_levels, mesh_same_rep, mesh_up_rep, up_gnns, up_same_gnns
            )

        # Note: We return all, even though only down edges really are used later
        return mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
