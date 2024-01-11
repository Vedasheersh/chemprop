from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch_geometric

from .mpn import MPN
from .ffn import build_ffn, MultiReadout
from chemprop.data import ProteinGraphDataset
from .gvp_models import GVPEmbedderModel
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import initialize_weights
from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
import ipdb

import torch
import torch.nn as nn

import ipdb
import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d
    
class AttentivePooling(nn.Module):
    def __init__(self, input_size=1280, hidden_size=1280):
        super(AttentivePooling, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Attention mechanism components
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        # Calculate attention weights
        attn_scores = self.linear2(self.tanh(self.linear1(input_tensor)))
        attn_weights = self.softmax(attn_scores)

        # Apply attention weights to the input tensor
        attn_applied = torch.bmm(attn_weights.permute(0, 2, 1), input_tensor)

        return attn_applied.squeeze(1), attn_weights.squeeze(-1)

class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()
        self.classification = args.dataset_type == "classification"
        self.multiclass = args.dataset_type == "multiclass"
        self.loss_function = args.loss_function
        self.args = args
        self.device = args.device

        if hasattr(args, "train_class_sizes"):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in [
                "cross_entropy",
                "binary_cross_entropy",
            ]

        self.is_atom_bond_targets = args.is_atom_bond_targets

        if self.is_atom_bond_targets:
            self.atom_targets, self.bond_targets = args.atom_targets, args.bond_targets
            self.atom_constraints, self.bond_constraints = (
                args.atom_constraints,
                args.bond_constraints,
            )
            self.adding_bond_types = args.adding_bond_types

        self.relative_output_size = 1
        if self.multiclass:
            self.relative_output_size *= args.multiclass_num_classes
        if self.loss_function == "mve":
            self.relative_output_size *= 2  # return means and variances
        if self.loss_function == "dirichlet" and self.classification:
            self.relative_output_size *= (
                2  # return dirichlet parameters for positive and negative class
            )
        if self.loss_function == "evidential":
            self.relative_output_size *= (
                4  # return four evidential parameters: gamma, lambda, alpha, beta
            )

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ["mve", "evidential", "dirichlet"]:
            self.softplus = nn.Softplus()

        self.create_encoder(args)
        
        print('Creating sequence model')
        self.create_sequence_model(args)
        self.create_ffn(args)

        initialize_weights(self)
        print('Creating graph sequence model')
        self.create_graph_sequence_model(args)

    def create_sequence_model(self, args: TrainArgs) -> None:
        """
        Creates the sequence model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.sequence_model = build_ffn(
                            first_linear_dim = args.gvp_node_hidden_dims[0] * args.gvp_num_layers,
                            hidden_size = args.protein_mlp_hidden_size,
                            num_layers = args.protein_mlp_num_layers,
                            output_size = args.protein_mlp_output_size,
                            dropout = args.protein_mlp_dropout,
                            activation = args.activation)
    
    def create_graph_sequence_model(self, args: TrainArgs) -> None:
        """
        Creates the sequence model with attention

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.protein_ds = lambda data_list: ProteinGraphDataset(data_list, 
                                                            include_esm_feats = args.use_esm_feats_gvp,
                                                            device=self.device)

        if not self.args.skip_gvp:
            self.gvp_model = GVPEmbedderModel(node_h_dim = args.gvp_node_hidden_dims,
                                             edge_h_dim = args.gvp_edge_hidden_dims,
                                             seq_embed_dim = args.gvp_seq_embed_dim,
                                             num_layers = args.gvp_num_layers,
                                             include_esm_feats = args.use_esm_feats_gvp)
        
        # attn_layer = AttentivePooling(dim, dim)
        # self.sequence_attention = attn_layer
        
    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
    def create_embed_model(self, args: TrainArgs) -> None:
        """
        Creates the embedding model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.embed_model = EmbedderModel(args)

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        second_linear_dim = None
        self.multiclass = args.dataset_type == "multiclass"
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent and not args.reaction_substrate:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            elif args.reaction_solvent and args.reaction_substrate:
                first_linear_dim = args.hidden_size
                second_linear_dim = args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == "descriptor":
            atom_first_linear_dim = first_linear_dim + args.atom_descriptors_size
        else:
            atom_first_linear_dim = first_linear_dim

        if args.bond_descriptors == "descriptor":
            bond_first_linear_dim = first_linear_dim + args.bond_descriptors_size
        else:
            bond_first_linear_dim = first_linear_dim

        # Create FFN layers
        if self.is_atom_bond_targets:
            self.readout = MultiReadout(
                atom_features_size=atom_first_linear_dim,
                bond_features_size=bond_first_linear_dim,
                atom_hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                bond_hidden_size=args.ffn_hidden_size + args.bond_descriptors_size,
                num_layers=args.ffn_num_layers,
                output_size=self.relative_output_size,
                dropout=args.dropout,
                activation=args.activation,
                atom_constraints=args.atom_constraints,
                bond_constraints=args.bond_constraints,
                shared_ffn=args.shared_atom_bond_ffn,
                weights_ffn_num_layers=args.weights_ffn_num_layers,
            )
        else:
            first_linear_dim_now = atom_first_linear_dim
            if not args.protein_records_path is None:
                first_linear_dim_now += args.gvp_node_hidden_dims[0] * args.gvp_num_layers
            self.readout = build_ffn(
                first_linear_dim=first_linear_dim_now,
                hidden_size=args.ffn_hidden_size + args.atom_descriptors_size,
                num_layers=args.ffn_num_layers,
                output_size=self.relative_output_size * args.num_tasks,
                dropout=args.dropout,
                activation=args.activation,
                dataset_type=args.dataset_type,
                spectra_activation=args.spectra_activation,
            )

        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                if self.is_atom_bond_targets:
                    if args.shared_atom_bond_ffn:
                        for param in list(self.readout.atom_ffn_base.parameters())[
                            0 : 2 * args.frzn_ffn_layers
                        ]:
                            param.requires_grad = False
                        for param in list(self.readout.bond_ffn_base.parameters())[
                            0 : 2 * args.frzn_ffn_layers
                        ]:
                            param.requires_grad = False
                    else:
                        for ffn in self.readout.ffn_list:
                            if ffn.constraint:
                                for param in list(ffn.ffn.parameters())[
                                    0 : 2 * args.frzn_ffn_layers
                                ]:
                                    param.requires_grad = False
                            else:
                                for param in list(ffn.ffn_readout.parameters())[
                                    0 : 2 * args.frzn_ffn_layers
                                ]:
                                    param.requires_grad = False
                else:
                    for param in list(self.readout.parameters())[
                        0 : 2 * args.frzn_ffn_layers
                    ]:  # Freeze weights and bias for given number of layers
                        param.requires_grad = False

    def fingerprint(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        fingerprint_type: str = "MPN",
    ) -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == "MPN":
            return self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "embed":
            return self.embed_model(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
        elif fingerprint_type == "last_FFN":
            return self.readout[1](
                self.readout(
                    batch,
                    features_batch,
                    atom_descriptors_batch,
                    atom_features_batch,
                    bond_descriptors_batch,
                    bond_features_batch,
                )
            )
        else:
            raise ValueError(f"Unsupported fingerprint type {fingerprint_type}.")

    def forward(
        self,
        batch: Union[
            List[List[str]],
            List[List[Chem.Mol]],
            List[List[Tuple[Chem.Mol, Chem.Mol]]],
            List[BatchMolGraph],
        ],
        features_batch: List[np.ndarray] = None,
        atom_descriptors_batch: List[np.ndarray] = None,
        atom_features_batch: List[np.ndarray] = None,
        bond_descriptors_batch: List[np.ndarray] = None,
        bond_features_batch: List[np.ndarray] = None,
        constraints_batch: List[torch.Tensor] = None,
        bond_types_batch: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_descriptors_batch: A list of numpy arrays containing additional bond descriptors.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :param constraints_batch: A list of PyTorch tensors which applies constraint on atomic/bond properties.
        :param bond_types_batch: A list of PyTorch tensors storing bond types of each bond determined by RDKit molecules.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions.
        """
        if self.is_atom_bond_targets:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )
            output = self.readout(encodings, constraints_batch, bond_types_batch)
        else:
            encodings = self.encoder(
                batch,
                features_batch,
                atom_descriptors_batch,
                atom_features_batch,
                bond_descriptors_batch,
                bond_features_batch,
            )

            if not self.args.protein_records_path is None:
                protein_records = batch[-1].protein_record_list
                protein_ds = self.protein_ds(protein_records)
                protein_dataloader = torch_geometric.data.DataLoader(protein_ds,
                                                                     batch_size=self.args.batch_size)
                for protein_batch in protein_dataloader:
                    protein_batch = protein_batch.to(self.device)
                    break

                protein_outs = self.gvp_model(protein_batch)
                # ipdb.set_trace()
                # sequence_feature_arr = pad_sequence(sequence_feature_arr,batch_first=True).to(self.device)
                # sequence_token_arr = pad_sequence(sequence_token_arr,
                #                                   padding_value=1,batch_first=True).to(self.device)
                # for self_attn_block in self.protein_self_attns:
                #     sequence_feature_arr = self_attn_block(sequence_feature_arr)
                
                # sequence_output = self.esm_model(sequence_token_arr,repr_layers=[len(self.esm_model.layers)])['representations'][len(self.esm_model.layers)][:, 1: len(sequence_feature_arr[0]) + 1, :]
                # sequence_output, sequence_weights = self.sequence_attention(torch.cat([sequence_feature_arr,sequence_output],dim=-1))
                # sequence_output, sequence_weights = self.sequence_attention(esm_if_out)
                
                # sequence_output = self.sequence_model(protein_outs)
                encodings = torch.concat([encodings,protein_outs],dim=-1)
            
            #print(self.readout)
            #print(encodings.shape)
            output = self.readout(encodings)
            # print(output.shape)

        # Don't apply sigmoid during training when using BCEWithLogitsLoss
        if (
            self.classification
            and not (self.training and self.no_training_normalization)
            and self.loss_function != "dirichlet"
        ):
            if self.is_atom_bond_targets:
                output = [self.sigmoid(x) for x in output]
            else:
                output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape(
                (output.shape[0], -1, self.num_classes)
            )  # batch size x num targets x num classes per target
            if (
                not (self.training and self.no_training_normalization)
                and self.loss_function != "dirichlet"
            ):
                output = self.multiclass_softmax(
                    output
                )  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        # Modify multi-input loss functions
        if self.loss_function == "mve":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, variances = torch.split(x, x.shape[1] // 2, dim=1)
                    variances = self.softplus(variances)
                    outputs.append(torch.cat([means, variances], axis=1))
                return outputs
            else:
                means, variances = torch.split(output, output.shape[1] // 2, dim=1)
                variances = self.softplus(variances)
                output = torch.cat([means, variances], axis=1)
        if self.loss_function == "evidential":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    means, lambdas, alphas, betas = torch.split(
                        x, x.shape[1] // 4, dim=1
                    )
                    lambdas = self.softplus(lambdas)  # + min_val
                    alphas = (
                        self.softplus(alphas) + 1
                    )  # + min_val # add 1 for numerical contraints of Gamma function
                    betas = self.softplus(betas)  # + min_val
                    outputs.append(torch.cat([means, lambdas, alphas, betas], dim=1))
                return outputs
            else:
                means, lambdas, alphas, betas = torch.split(
                    output, output.shape[1] // 4, dim=1
                )
                lambdas = self.softplus(lambdas)  # + min_val
                alphas = (
                    self.softplus(alphas) + 1
                )  # + min_val # add 1 for numerical contraints of Gamma function
                betas = self.softplus(betas)  # + min_val
                output = torch.cat([means, lambdas, alphas, betas], dim=1)
        if self.loss_function == "dirichlet":
            if self.is_atom_bond_targets:
                outputs = []
                for x in output:
                    outputs.append(nn.functional.softplus(x) + 1)
                return outputs
            else:
                output = nn.functional.softplus(output) + 1

        return output

