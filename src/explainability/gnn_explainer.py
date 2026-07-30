"""
GNNExplainer for molecular nephrotoxicity prediction.

Explains important atoms and molecular substructures
from trained GIN model predictions.
"""

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, MaskType


def create_gnn_explainer(model):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        model_config=ModelConfig(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object,
    )

    return explainer


def explain_graph(explainer, data):

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index
    )

    node_importance = explanation.node_mask.detach().cpu()

    return node_importance
