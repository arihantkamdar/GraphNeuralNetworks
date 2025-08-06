# Project Overview

This repository, GraphNeuralNetworks, is a collection of implementations and experiments focused on Graph Neural Networks (GNNs) for solving graph-based machine learning problems. It includes code for training and evaluating GNN models, such as Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), on tasks like node classification and link prediction. The project is designed for researchers and developers interested in exploring GNN applications using PyTorch and PyTorch Geometric.


Features





- Implementation of GNN models namely, GCN, GAT, GCN-LSTM, STGCN and MTGNN Networks.



- Support for Financial Data .



- Modular code structure for easy experimentation with custom GNN architectures.



- Scripts for preprocessing graph data and evaluating model performance.



## Installation

To set up the project environment, follow these steps:
```bash
# 1. Clone the repository
git clone [repo](https://github.com/arihantkamdar/GraphNeuralNetworks.git/)
cd repo

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
# if this creates an issue, use poetry! 


# 3. Install dependencies
pip install -r r.txt

```
Each Model has a structure like:
- Data.py: Which helps in preprocessing CSV data into graph structures
- Model.py: Where the core Deep learning model is
- train.py: The code to run training and inference pipeline: here all the parameters and hyperparamters are defined

To run  the project or a particular model:

- select the model you want to run and go in that directory
- ```bashrun python3 train.py





