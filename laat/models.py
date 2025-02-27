# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# from abc import ABC, abstractmethod
# from typing import Optional
# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# import numpy as np


# from skorch.dataset import unpack_data
# from skorch.regressor import NeuralNetRegressor
# from skorch.classifier import NeuralNetBinaryClassifier


# class LAATRegressor(NeuralNetRegressor):
#     def __init__(self, *args, gamma, llm_ratings, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.gamma = gamma
#         self.llm_ratings = llm_ratings

#     def train_step_single(self, batch, **fit_params):
#         self._set_training(True)
#         Xi, yi = unpack_data(batch)
#         Xi.requires_grad = True
#         y_pred = self.infer(Xi, **fit_params)
#         loss = self.get_loss(y_pred, yi, X=Xi, training=True)

#         # calculate attribution loss
#         o_grad = torch.autograd.grad(y_pred.sum(), Xi, create_graph=True, retain_graph=True)[0]
#         scaled_llm_ratings = F.normalize(self.llm_ratings, dim=-1).to(o_grad.device) * o_grad.norm(dim=-1, keepdim=True)
#         attributions = o_grad * Xi
#         att_loss = nn.MSELoss()(attributions, (scaled_llm_ratings * Xi).detach())

#         loss += self.gamma * att_loss
#         loss.backward()
#         return {
#             "loss": loss,
#             "y_pred": y_pred,
#         }

#     def validation_step(self, batch, **fit_params):
#         self._set_training(False)
#         Xi, yi = unpack_data(batch)
#         Xi.requires_grad = True
#         y_pred = self.infer(Xi, **fit_params)
#         loss = self.get_loss(y_pred, yi, X=Xi, training=False)
#         # calculate attribution loss
#         o_grad = torch.autograd.grad(y_pred.sum(), Xi)[0]
#         scaled_llm_ratings = F.normalize(self.llm_ratings, dim=-1).to(o_grad.device) * o_grad.norm(dim=-1, keepdim=True)
#         attributions = o_grad * Xi
#         att_loss = nn.MSELoss()(attributions, (scaled_llm_ratings * Xi).detach())

#         loss += self.gamma * att_loss

#         return {
#             "loss": loss,
#             "y_pred": y_pred,
#         }


# class LAATClassifier(NeuralNetBinaryClassifier):
#     def __init__(self, *args, gamma, is_mse, llm_ratings, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.gamma = gamma
#         self.is_mse = is_mse
#         self.llm_ratings = llm_ratings

#     def __str__(self):
#         return f"{self.__class__.__name__}_{self.gamma}"

#     def train_step_single(self, batch, **fit_params):
#         self._set_training(True)
#         Xi, yi = unpack_data(batch)
#         Xi.requires_grad = True
#         y_pred = self.infer(Xi, **fit_params)
#         loss = self.get_loss(y_pred, yi, X=Xi, training=True)

#         # calculate attribution loss
#         attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
#         llm_attributions = F.normalize(self.llm_ratings.to(attributions.device) * Xi, dim=-1) * attributions.norm(
#             dim=-1, keepdim=True
#         )

#         # att_loss = nn.MSELoss()(attributions, llm_attributions.detach())
#         if self.is_mse:
#             att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_attributions, dim=-1).detach())
#         else:
#             att_loss = (1 - (F.cosine_similarity(attributions, llm_attributions.detach()) * 2 - 1)).mean()

#         loss += self.gamma * att_loss
#         loss.backward()

#         return {
#             "loss": loss,
#             "y_pred": y_pred,
#         }

#     def validation_step(self, batch, **fit_params):
#         self._set_training(False)
#         Xi, yi = unpack_data(batch)
#         Xi.requires_grad = True
#         y_pred = self.infer(Xi, **fit_params)
#         loss = self.get_loss(y_pred, yi, X=Xi, training=False)

#         # calculate attribution loss
#         attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
#         llm_attributions = F.normalize(self.llm_ratings.to(attributions.device) * Xi, dim=-1) * attributions.norm(
#             dim=-1, keepdim=True
#         )

#         # att_loss = nn.MSELoss()(attributions, llm_attributions.detach())
#         # att_loss = (1 - (F.cosine_similarity(attributions, llm_attributions.detach()) * 2 - 1)).mean()
#         att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_attributions, dim=-1).detach())

#         loss += self.gamma * att_loss

#         return {
#             "loss": loss,
#             "y_pred": y_pred,
#         }


# class LAATModel(ABC):
#     def fit(
#         self,
#         X: torch.tensor,
#         y: torch.tensor,
#         gamma: float,  # attribution loss weight
#         llm_ratings: torch.tensor,
#         nsteps: int = 1000,
#         batch_size: int = 128,
#         seed: int = 69,
#         device: str = "cpu",
#     ) -> None:
#         optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
#         step_counter = 0

#         train_dataset = TensorDataset(X, y)
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         llm_ratings = llm_ratings.to(device)

#         torch.manual_seed(seed)
#         self.model.train()
#         while step_counter < nsteps:
#             for x_batch, y_batch in train_loader:
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 optimizer.zero_grad()

#                 x_batch.requires_grad = True

#                 o = self.model(x_batch)

#                 o_grad = torch.autograd.grad(o.sum(), x_batch, create_graph=True, retain_graph=True)[0]
#                 scaled_llm_ratings = F.normalize(llm_ratings, dim=-1) * o_grad.norm(dim=-1, keepdim=True)
#                 attributions = o_grad * x_batch

#                 weight = torch.where(y_batch == 1, 1.0, y.mean() / (1 - y.mean()))  # weighted loss

#                 cls_loss = nn.BCEWithLogitsLoss(reduction="none")(o, y_batch[:, None]).flatten() @ weight
#                 att_loss = nn.MSELoss()(attributions, (scaled_llm_ratings * x_batch).detach())
#                 loss = cls_loss + gamma * att_loss
#                 loss.backward()
#                 optimizer.step()

#                 step_counter += 1

#                 if step_counter >= nsteps:
#                     break

#     def score(self, X: torch.tensor, y: torch.tensor, device: str = "cpu") -> tuple[float, float, float]:
#         self.model.eval()
#         with torch.no_grad():
#             y_p = self.model(X.to(device)).sigmoid().detach().cpu().numpy()
#             y_pred = (y_p > 0.5).astype(int)
#             y_pred_proba = y_p

#         return accuracy_score(y, y_pred), f1_score(y, y_pred), roc_auc_score(y, y_pred_proba)


# class LogisticRegression(LAATModel):
#     def __init__(self):
#         self.model = nn.LazyLinear(1)


# class MLP(LAATModel):
#     def __init__(self, n_hidden: list[int]):
#         layers = [l for nh in n_hidden for l in [nn.LazyLinear(nh), nn.ReLU()]] + [nn.LazyLinear(1)]
#         self.model = nn.Sequential(*layers)


# class SklearnClassifier(LAATModel):
#     def __init__(self, classifier_class, **kwargs):
#         self.model = classifier_class(**kwargs)

#     def fit(
#         self,
#         X: torch.tensor,
#         y: torch.tensor,
#         gamma: float,  # attribution loss weight
#         llm_ratings: torch.tensor,
#         nsteps: int = 1000,
#         batch_size: int = 128,
#         seed: int = 69,
#         device: str = "cpu",
#     ) -> None:
#         self.model.fit(X, y)

#     def score(self, X: torch.tensor, y: torch.tensor, device: str = "cpu") -> tuple[float, float, float]:
#         y_p = self.model.predict_proba(X)[:, 1]
#         y_pred = (y_p > 0.5).astype(int)
#         y_pred_proba = y_p

#         return accuracy_score(y, y_pred), f1_score(y, y_pred), roc_auc_score(y, y_pred_proba)
