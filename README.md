<p align="center">
<img width="350" alt="License: GPL--3.0" src="laat_final_logo.png" />
<h1 align="center"><b>L</b>arge Language Model <b>A</b>ttribution <b>A</b>ligned <b>T</b>raining (<b>LAAT</b>)</h1>

</p>
<p>
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html" target="_blank">
    <img alt="License: GPL--3.0" src="https://img.shields.io/badge/License-GPL--3.0-yellow.svg" />
  </a>
</p>

Use LLMs as training regularizers for small models and significantly improve their generalization ability. Read more in our paper: [Large Language Models as Attribution Regularizers for Efficient Model Training](https://arxiv.org/abs/2502.20268)


## Quickstart 🚀

You can quickly train a model on a specified dataset using LLM attribution guidance:

```python
from laat.datasets import LAATDataset
from laat.splitters import NShotSplitter
from laat.models.laat import LAATLAATModel, LAATClassifier, TorchLogisticRegression
from langchain_openai import ChatOpenAI


# load the dataset
dataset = LAATDataset("breast-ljubljana", "laat/data")
# split it into k-shot
X_train, X_test, y_train, y_test = NShotSplitter.split(dataset.X, dataset.y, shot=5)

# define training parameters
model_kwargs = {
    "lr": 1e-2,
    "max_epochs": 200,
    "train_split": None,
    "optimizer": torch.optim.Adam,
    "optimizer__weight_decay": 1e-2,
    "verbose": False,
    "device": "cuda",
}

# instantiate the model
model = LAATLAATModel(
            model_name=f"laat_gpt-4o-mini_lr",
            model_class=partial_class(
                LAATClassifier,
                module=TorchLogisticRegression,
                **model_kwargs,
                ),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
            parsing_llm=ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.0,
                ),
            gamma=100.0,
            n_estimates=5,
        )

# train the model
model.train(X_train, y_train)
```

To train a model you need a **.csv** dataset, and a metadata **.json** file describing the task and listing the descriptions of all features. You can define a metadata file manually, or generate one automatically, by providing a dataset and the task description:

```python
from laat.datasets import LAATDataset
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-nano")

LAATDataset.generate_metadata(
    dataset_name="indian_liver",
    dataset_task_description="Predict whether the patient has a liver disease. Yes or no?",
    model=model,
    data_root="laat/data",
)
```

## Citation

If you find this repository or the paper useful in your research, please cite us using the following BibTeX entry:
```text
@misc{vukadin2025largelanguagemodelsattribution,
      title={Large Language Models as Attribution Regularizers for Efficient Model Training}, 
      author={Davor Vukadin and Marin Šilić and Goran Delač},
      year={2025},
      eprint={2502.20268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.20268}, 
}
```

## Author

👤 **Davor Vukadin**

* Github: [@davor10105](https://github.com/davor10105)
* LinkedIn: [@https:\/\/www.linkedin.com\/in\/davor-vukadin-596aaa1b7\/](https://linkedin.com/in/https:\/\/www.linkedin.com\/in\/davor-vukadin-596aaa1b7\/)
