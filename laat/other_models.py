# The SAINT model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from torch import einsum
from einops import rearrange


"""
    SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
    (https://arxiv.org/abs/2106.01342)
    
    Code adapted from: https://github.com/somepago/saint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import os

import numpy as np
import typing as tp

import optuna

import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange


import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch
import numpy as np


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == "MLP":
        x_cont_enc = torch.empty(n1, n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    else:
        raise Exception("This case should not work!")

    x_cont_enc = x_cont_enc.to(device)
    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)

    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

    if vision_dset:
        pos = np.tile(np.arange(x_categ.shape[-1]), (x_categ.shape[0], 1))
        pos = torch.from_numpy(pos).to(device)
        pos_enc = model.pos_encodings(pos)
        x_categ_enc += pos_enc

    return x_categ, x_categ_enc, x_cont_enc


def mixup_data(x1, x2, lam=1.0, y=None, use_cuda=True):
    """Returns mixed inputs, pairs of targets"""

    batch_size = x1.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
        return mixed_x1, mixed_x2, y_a, y_b

    return mixed_x1, mixed_x2


def add_noise(x_categ, x_cont, noise_params={"noise_type": ["cutmix"], "lambda": 0.1}):
    lam = noise_params["lambda"]
    device = x_categ.device
    batch_size = x_categ.size()[0]

    if "cutmix" in noise_params["noise_type"]:
        index = torch.randperm(batch_size)
        cat_corr = torch.from_numpy(np.random.choice(2, (x_categ.shape), p=[lam, 1 - lam])).to(device)
        con_corr = torch.from_numpy(np.random.choice(2, (x_cont.shape), p=[lam, 1 - lam])).to(device)
        x1, x2 = x_categ[index, :], x_cont[index, :]
        x_categ_corr, x_cont_corr = x_categ.clone().detach(), x_cont.clone().detach()
        x_categ_corr[cat_corr == 0] = x1[cat_corr == 0]
        x_cont_corr[con_corr == 0] = x2[con_corr == 0]
        return x_categ_corr, x_cont_corr
    elif noise_params["noise_type"] == "missing":
        x_categ_mask = np.random.choice(2, (x_categ.shape), p=[lam, 1 - lam])
        x_cont_mask = np.random.choice(2, (x_cont.shape), p=[lam, 1 - lam])
        x_categ_mask = torch.from_numpy(x_categ_mask).to(device)
        x_cont_mask = torch.from_numpy(x_cont_mask).to(device)
        return torch.mul(x_categ, x_categ_mask), torch.mul(x_cont, x_cont_mask)

    else:
        print("yet to write this")


def data_split(X, y, nan_mask):  # indices
    x_d = {"data": X, "mask": nan_mask.values}

    if x_d["data"].shape != x_d["mask"].shape:
        raise "Shape of data not same as that of nan mask!"

    y_d = {"data": y.reshape(-1, 1)}
    return x_d, y_d


def data_prep(X, y):
    temp = pd.DataFrame(X).fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    X, y = data_split(X, y, nan_mask)
    return X, y


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task="regression", continuous_mean_std=None):

        X_mask = X["mask"].copy()
        X = X["data"].copy()

        # Added this to handle data without categorical features
        if cat_cols is not None:
            cat_cols = list(cat_cols)
            con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        else:
            con_cols = list(np.arange(X.shape[1]))
            cat_cols = []

        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        if task == "regression":
            self.y = Y["data"].astype(np.float32)
        else:
            self.y = Y["data"]  # .astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return (
            np.concatenate((self.cls[idx], self.X1[idx])),
            self.X2[idx],
            self.y[idx],
            np.concatenate((self.cls_mask[idx], self.X1_mask[idx])),
            self.X2_mask[idx],
        )


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2.0 * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style="col"):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == "colrow":
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))
                            ),
                            PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                            PreNorm(
                                dim * nfeats,
                                Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout)),
                            ),
                            PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim * nfeats,
                                Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout)),
                            ),
                            PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                        ]
                    )
                )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        if self.style == "colrow":
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2]))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


# main class


class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=1,
        continuous_mean_std=None,
        attn_dropout=0.0,
        ff_dropout=0.0,
        lastmlp_dropout=0.0,
        cont_embeddings="MLP",
        scalingfactor=10,
        attentiontype="col",
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), "number of each category must be positive"

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("categories_offset", categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == "MLP":
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print("Continous features are not passed through attention")
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("cat_mask_offset", cat_mask_offset)
        self.register_buffer("con_mask_offset", con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)

    def forward(self, x_categ, x_cont, x_categ_enc, x_cont_enc):
        device = x_categ.device
        if self.attentiontype == "justmlp":
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim=-1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == "MLP":
                x = self.transformer(x_categ_enc, x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else:
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim=-1)
        flat_x = x.flatten(1)
        return self.mlp(flat_x)


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class SAINTModel(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=0,
        attn_dropout=0.0,
        ff_dropout=0.0,
        cont_embeddings="MLP",
        scalingfactor=10,
        attentiontype="col",
        final_mlp_style="common",
        y_dim=2,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), "number of each category must be positive"

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("categories_offset", categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == "MLP":
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == "pos_singleMLP":
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print("Continous features are not passed through attention")
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("cat_mask_offset", cat_mask_offset)
        self.register_buffer("con_mask_offset", con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        if self.final_mlp_style == "common":
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            self.mlp2 = simple_MLP([dim, (self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, self.num_continuous, np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )
        self.pt_mlp2 = simple_MLP(
            [
                dim * (self.num_continuous + self.num_categories),
                6 * dim * (self.num_continuous + self.num_categories) // 5,
                dim * (self.num_continuous + self.num_categories) // 2,
            ]
        )

    def forward(self, x_categ, x_cont):

        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:, : self.num_categories, :])
        con_outs = self.mlp2(x[:, self.num_categories :, :])
        return cat_outs, con_outs


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.
    At least they have to extend the init method defining the model and the define_trial_parameters method
    specifying the hyperparameters.

    Methods
    -------
    __init__(params, args):
        Defines the model architecture, depending on the hyperparameters (params) and command line arguments (args).
    fit(X, y, X_val=None, y_val=None)
        Trains the model on the trainings dataset (X, y). Validates the training process and uses early stopping
        if a validation set (X_val, y_val) is provided. Returns the loss history and validation loss history.
    predict(X)
        Predicts the labels of the test dataset (X). Saves and returns the predictions.
    attribute(X, y)
        Extract feature attributions for input pair (X, y)
    define_trial_parameters(trial, args)
        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization.
    save_model_and_prediction(y_true, filename_extension="")
        Saves the current state of the model and the predictions and true labels of the test dataset.
    save_model(filename_extension="")
        Saves the current state of the model.
    save_predictions(y_true, filename_extension="")
        Saves the predictions and true labels of the test dataset.
    clone()
        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.
    """

    def __init__(self, params: tp.Dict, args):
        """Defines the model architecture.

        After calling this method, self.model has to be defined.

        :param params: possible hyperparameter configuration, model architecture depends on this
        :param args: command line arguments containing all important information about the dataset and training process
        """
        self.args = args
        self.params = params

        # Model definition has to be implemented by the concrete model
        self.model = None

        # Create a placeholder for the predictions on the test dataset
        self.predictions = None
        self.prediction_probabilities = None  # Only used by binary / multi-class-classification

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: tp.Union[None, np.ndarray] = None,
        y_val: tp.Union[None, np.ndarray] = None,
    ) -> tp.Tuple[list, list]:
        """Trains the model.

        The training is done on the trainings dataset (X, y). If a validation set (X_val, y_val) is provided,
        the model state is validated during the training, to allow early stopping.

        Returns the loss history and validation loss history if the loss and validation loss development during
        the training are logged. Otherwise empty lists are returned.

        :param X: trainings data
        :param y: labels of trainings data
        :param X_val: validation data
        :param y_val: labels of validation data
        :return: loss history, validation loss history
        """

        self.model.fit(X, y)

        # Should return loss history and validation loss history
        return [], []

    def predict(self, X: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
        """
        Returns the regression value or the concrete classes of binary / multi-class-classification tasks.
        (Save predictions to self.predictions)

        :param X: test data
        :return: predicted values / classes of test data (Shape N x 1)
        """

        if self.args.objective == "regression":
            self.predictions = self.model.predict(X)
        elif self.args.objective == "classification" or self.args.objective == "binary":
            self.prediction_probabilities = self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Only implemented for binary / multi-class-classification tasks.
        Returns the probability distribution over the classes C.
        (Save probabilities to self.prediction_probabilities)

        :param X: test data
        :return: probabilities for the classes (Shape N x C)
        """

        self.prediction_probabilities = self.model.predict_proba(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if self.prediction_probabilities.shape[1] == 1:
            self.prediction_probabilities = np.concatenate(
                (1 - self.prediction_probabilities, self.prediction_probabilities), 1
            )
        return self.prediction_probabilities

    def save_model_and_predictions(self, y_true: np.ndarray, filename_extension=""):
        """Saves the current state of the model and the predictions and true labels of the test dataset.

        :param y_true: true labels of the test data
        :param filename_extension: (optional) additions to the filenames
        """
        self.save_predictions(y_true, filename_extension)
        self.save_model(filename_extension)

    def clone(self):
        """Clone the model.

        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.

        :return: Copy of the current model without trained parameters
        """
        return self.__class__(self.params, self.args)

    @classmethod
    def define_trial_parameters(cls, trial: optuna.Trial, args) -> tp.Dict:
        """Define the ranges of the hyperparameters

        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization. All hyperparameter that should be optimized and their ranges are specified here.
        For more information see: https://optuna.org/

        :param trial: Trial class instance generated by the optuna library.
        :param args: Command line arguments containing all important information about the dataset
        :return: Hyperparameter configuration
        """

        raise NotImplementedError("This method has to be implemented by the sub class")

    def save_model(self, filename_extension=""):
        """Saves the current state of the model.

        Saves the model using pickle. Override this method if model should be saved in a different format.

        :param filename_extension: true labels of the test data
        """
        print("called save model")
        # save_model_to_file(self.model, self.args, filename_extension)

    def save_predictions(self, y_true: np.ndarray, filename_extension=""):
        """Saves the predictions and true labels of the test dataset.

        Saves the predictions and the truth values together in a npy file.

        :param y_true: true labels of the test data
        :param filename_extension: true labels of the test data
        """
        if self.args.objective == "regression":
            # Save array where [:,0] is the truth and [:,1] the prediction
            y = np.concatenate((y_true.reshape(-1, 1), self.predictions.reshape(-1, 1)), axis=1)
        else:
            # Save array where [:,0] is the truth and [:,1:] are the prediction probabilities
            y = np.concatenate((y_true.reshape(-1, 1), self.prediction_probabilities), axis=1)
        print("called save prediction")
        # save_predictions_to_file(y, self.args, filename_extension)

    def get_model_size(self):
        raise NotImplementedError("Calculation of model size has not been implemented for this model.")

    def attribute(cls, X: np.ndarray, y: np.ndarray, strategy: str = "") -> np.ndarray:
        """Get feature attributions for inherently interpretable models. This function is only implemented for
        interpretable models.

        :param X: data (Shape N x D)
        :param y: labels (Shape N) for which the attribution should be computed for (
        usage of these labels depends on the specific model)

        :strategy: if there are different strategies that can be used to compute the attributions they can be passed
        here. Passing an empty sting should always result in the default strategy.

        :return The (non-normalized) importance attributions for each feature in each data point. (Shape N x D)
        """
        raise NotImplementedError(f"This method is not implemented for class {type(cls)}.")


class BaseModelTorch(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.device = self.get_device()
        self.gpus = args.gpu_ids if args.use_gpu and torch.cuda.is_available() and args.data_parallel else None

    def to_device(self):
        if self.args.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        print("On Device:", self.device)
        self.model.to(self.device)

    def get_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.data_parallel:
                device = "cuda"  # + ''.join(str(i) + ',' for i in self.args.gpu_ids)[:-1]
            else:
                device = "cuda"
        else:
            device = "cpu"

        return torch.device(device)

    def fit(self, X, y, X_val=None, y_val=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                out = self.model(batch_val_X.to(self.device))

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def predict(self, X):
        if self.args.objective == "regression":
            self.predictions = self.predict_helper(X)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X)

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=2
        )
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def save_model(self, filename_extension="", directory="models"):
        # filename = get_output_path(
        #     self.args, directory=directory, filename="m", extension=filename_extension, file_type="pt"
        # )
        # print("save_model")
        # print(filename)
        # torch.save(self.model.state_dict(), filename)
        print("save model called")

    def load_model(self, filename_extension="", directory="models"):
        # filename = get_output_path(
        #     self.args, directory=directory, filename="m", extension=filename_extension, file_type="pt"
        # )
        # print("load model")
        # print(filename)
        # state_dict = torch.load(filename)
        # os.remove(filename)
        # self.model.load_state_dict(state_dict)
        print("call model called")

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    @classmethod
    def define_trial_parameters(cls, trial, args):
        raise NotImplementedError("This method has to be implemented by the sub class")


class SAINT(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.model_id = args.model_id
        if len(args.cat_idx) > 0:
            num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            # Appending 1 for CLS token, this is later used to generate embeddings.
            cat_dims = np.append(np.array([1]), np.array(args.cat_dims)).astype(int)
        else:
            num_idx = list(range(args.num_features))
            cat_dims = np.array([1])

        # Decreasing some hyperparameter to cope with memory issues
        dim = self.params["dim"] if args.num_features < 50 else 8
        self.batch_size = self.args.batch_size if args.num_features < 50 else 64

        print("Using dim %d and batch size %d" % (dim, self.batch_size))

        self.model = SAINTModel(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=dim,
            dim_out=1,
            depth=self.params["depth"],  # 6
            heads=self.params["heads"],  # 8
            attn_dropout=self.params["dropout"],  # 0.1
            ff_dropout=self.params["dropout"],  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=args.num_classes,
        )

        # if self.args.data_parallel:
        #     self.model.transformer = nn.DataParallel(self.model.transformer, device_ids=self.args.gpu_ids)
        #     self.model.mlpfory = nn.DataParallel(self.model.mlpfory, device_ids=self.args.gpu_ids)

    def fit(self, X, y, X_val=None, y_val=None):
        # X_val = X[:int(0.2 * len(X))]
        # y_val = y[:int(0.2 * len(y))]
        # X = X[int(0.2 * len(X)):]
        # y = y[int(0.2 * len(y)):]

        if self.args.objective == "binary":
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.objective == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)  # lr=0.00003)

        self.model.to(self.device)

        # SAINT wants it like this...
        X = {"data": X, "mask": np.ones_like(X)}
        y = {"data": y.reshape(-1, 1)}

        if X_val is not None:
            X_val = {"data": X_val, "mask": np.ones_like(X_val)}
            y_val = {"data": y_val.reshape(-1, 1)}

        train_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

        if X_val is not None:
            val_ds = DataSetCatCon(X_val, y_val, self.args.cat_idx, self.args.objective)
            valloader = DataLoader(val_ds, batch_size=self.args.val_batch_size, shuffle=True, num_workers=2)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            self.model.train()

            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()

                # x_categ is the the categorical data,
                # x_cont has continuous data,
                # y_gts has ground truth ys.
                # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
                # token) set to 0s.
                # con_mask is an array of ones same shape as x_cont.
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                # We are converting the data to embeddings in the next step
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # select only the representations corresponding to CLS token
                # and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:, 0, :]

                y_outs = self.model.mlpfory(y_reps)

                if self.args.objective == "regression":
                    y_gts = y_gts.to(self.device)
                elif self.args.objective == "classification":
                    y_gts = y_gts.to(self.device).squeeze()
                else:
                    y_gts = y_gts.to(self.device).float()

                loss = criterion(y_outs, y_gts)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                # print("Loss", loss.item())

            # Early Stopping
            if X_val is not None:
                val_loss = 0.0
                val_dim = 0
                self.model.eval()
                with torch.no_grad():
                    for data in valloader:
                        x_categ, x_cont, y_gts, cat_mask, con_mask = data

                        x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                        cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                        reps = self.model.transformer(x_categ_enc, x_cont_enc)
                        y_reps = reps[:, 0, :]
                        y_outs = self.model.mlpfory(y_reps)

                        if self.args.objective == "regression":
                            y_gts = y_gts.to(self.device)
                        elif self.args.objective == "classification":
                            y_gts = y_gts.to(self.device).squeeze()
                        else:
                            y_gts = y_gts.to(self.device).float()

                        val_loss += criterion(y_outs, y_gts)
                        val_dim += 1
                val_loss /= val_dim

                val_loss_history.append(val_loss.item())

                print("Epoch", epoch, "loss", val_loss.item())

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_idx = epoch

                    # Save the currently best model
                    self.save_model(filename_extension="{}_best".format(self.model_id), directory="tmp")

                if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                    print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                    print("Early stopping applies.")
                    break

        # self.load_model(filename_extension="{}_best".format(self.model_id), directory="tmp")
        return loss_history, val_loss_history

    def predict_helper(self, X):
        X = {"data": X, "mask": np.ones_like(X)}
        y = {"data": np.ones((X["data"].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()

        predictions = []

        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                if self.args.objective == "binary":
                    y_outs = torch.sigmoid(y_outs)
                elif self.args.objective == "classification":
                    y_outs = F.softmax(y_outs, dim=1)

                predictions.append(y_outs.detach().cpu().numpy())
        return np.concatenate(predictions)

    def attribute(self, X, y, strategy=""):
        """Generate feature attributions for the model input.
        Two strategies are supported: default ("") or "diag". The default strategie takes the sum
        over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
        of the attention map.
        return array with the same shape as X.
        """
        global my_attention
        # self.load_model(filename_extension="best", directory="tmp")

        X = {"data": X, "mask": np.ones_like(X)}
        y = {"data": np.ones((X["data"].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()
        # print(self.model)
        # Apply hook.
        my_attention = torch.zeros(0)

        def sample_attribution(layer, minput, output):
            global my_attention
            # print(minput)
            """ an hook to extract the attention maps. """
            h = layer.heads
            q, k, v = layer.to_qkv(minput[0]).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
            sim = einsum("b h i d, b h j d -> b h i j", q, k) * layer.scale
            my_attention = sim.softmax(dim=-1)

        # print(type(self.model.transformer.layers[0][0].fn.fn))
        self.model.transformer.layers[0][0].fn.fn.register_forward_hook(sample_attribution)
        attributions = []
        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                # print(x_categ.shape, x_cont.shape)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                # y_reps = reps[:, 0, :]
                # y_outs = self.model.mlpfory(y_reps)
                if strategy == "diag":
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].diagonal(0, 1, 2))
                else:
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].sum(dim=1))

        attributions = np.concatenate(attributions)
        return attributions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "dim": trial.suggest_categorical("dim", [32, 64, 128, 256]),
            "depth": trial.suggest_categorical("depth", [1, 2, 3, 6, 12]),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        }
        return params
