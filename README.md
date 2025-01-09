
# Classification of Wikipedia pages 

This repository was carried out by group 3 in [the MLOps course at DTU](https://skaftenicki.github.io/dtu_mlops/). 
Group 3 consists of: Clara Regine Hoeg Kold, Rasmus Lyhne Fiehn, Emma Christine Berggrein Andersen, Frederik Baj Henriksen, and Ollie Elmgreen.


## Overall goal of the project
The objective of this project is to classifiy the Wiki-CS Dataset into its 10 native classes (Wikipedia topics) using the Pytorch Geometric framework. 

## Framework
The framework used is the [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io), which implements neural network layers for graphs. 

## Data
In this project, the [Wiki-CS Dataset](https://github.com/pmernyei/wiki-cs-dataset) will be used. This dataset consists of 11,701 Wikipedia pages represented as nodes in the graph, and 216,123 edges representing hyperlinks between the pages. The Wikipedia pages are split into 10 topic classes. 

## Models
In this implementation, Graph Neural Network (GNN) models with various Graph Convolutional Network (GCN) layers is utilized. 


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
