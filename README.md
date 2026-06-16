# MIP

Official implementation of the MIP paper:

- QiAo Yuan, Boxuan Zhu, Weizhi Huang, Sheng-Uei Guan, Ka Lok Man, "MIP: Mutual information-guided prompt for class-incremental continual graph learning", Pattern Recognition, 2026.
- Paper link: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320326006692?fr=RR-2&ref=pdf_download&rr=a0c6bef11f5f2ad3)


## Repository Layout

- `train.py`: main entry point for hyper-parameter search, validation, and testing.
- `pipeline.py`: experiment pipeline for class-IL and task-IL settings.
- `Backbones/`: backbone GNN implementations and dataset loading helpers.
- `Baselines/`: continual learning methods, including the MIP-related models.
- `dataset/`: task manager utilities.
- `training/`: shared utility functions.
- `visualize.py`: result summarization and visualization helpers.

## Environment

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Some packages such as `dgl`, `torch`, and `torch-geometric` can require environment-specific installation depending on your CUDA version. If needed, install the matching build for your machine first, then install the remaining packages from `requirements.txt`.

## Data Preparation

This code uses two kinds of dataset sources:

- `Arxiv-CL` and `Products-CL` are based on OGB node property prediction datasets (`ogbn-arxiv` and `ogbn-products`).
- `CoraFull-CL`, `Citeseer`, `RomanEmpire`, and `Reddit-CL` can be downloaded directly through DGL dataset classes.

For the OGB datasets, please refer to the official dataset pages:

- `ogbn-arxiv`: [OGB Node Property Prediction docs](https://ogb.stanford.edu/docs/nodeprop/)
- `ogbn-products`: [OGB Node Property Prediction docs](https://ogb.stanford.edu/docs/nodeprop/)

For the DGL built-in datasets, please refer to:

- `CoraFullDataset`: [DGL CoraFullDataset docs](https://www.dgl.ai/dgl_docs/generated/dgl.data.CoraFullDataset.html)
- `CiteseerGraphDataset`: [DGL CiteseerGraphDataset docs](https://www.dgl.ai/dgl_docs/generated/dgl.data.CiteseerGraphDataset.html)
- `RomanEmpireDataset`: [DGL RomanEmpireDataset docs](https://www.dgl.ai/dgl_docs/generated/dgl.data.RomanEmpireDataset.html)
- `RedditDataset`: [DGL RedditDataset docs](https://www.dgl.ai/dgl_docs/generated/dgl.data.RedditDataset.html)

The DGL `RedditDataset` page states that this dataset is built from Reddit posts from September 2014 for community detection / node classification, and cites the GraphSAGE dataset source:

- GraphSAGE Reddit reference: [SNAP / GraphSAGE data page](http://snap.stanford.edu/graphsage/)

When running the code, set `--ori_data_path` to the directory where your raw datasets are stored. For example, in our training setup we use:

```bash
--ori_data_path /mnt/datasets
```

The repository may also create processed task splits under `./data` during preprocessing.

## Running Experiments

Example:

```bash
python train.py --dataset Arxiv-CL --method mip --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
```

Common arguments:

- `--dataset`: dataset name such as `Reddit-CL`, `Products-CL`, `Arxiv-CL`, or `CoraFull-CL`
- `--method`: one of `mip`, `tpp`, `ergnn`, `twp`, or `ewc`
- `--backbone`: backbone GNN
- `--ILmode`: `taskIL` or `classIL`
- `--gpu`: GPU index
- `--inter-task-edges`: whether to keep edges across tasks
- `--minibatch`: whether to use mini-batch training
- `--ori_data_path`: root directory of the raw datasets on your machine

## Notes

- `train.py` is the maintained experiment entry point.
- The repository currently assumes local availability of the required datasets.
- The command above matches the main class-incremental MIP setting used in our experiments.
- The cleanup avoids changing the core algorithmic design or method-specific training behavior.

## Acknowledgements

Parts of the implementation were developed with reference to the following repositories:

- [mala-lab/TPP](https://github.com/mala-lab/TPP)
- [QueuQ/CGLB](https://github.com/QueuQ/CGLB/tree/master)
