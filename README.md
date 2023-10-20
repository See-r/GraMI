# GraMI
This is the official repository of paper **Variational Graph Autoencoders for Heterogeneous Information Networks with Missing and Inaccurate Attributes**.

# Datasets
Our datasets are downloaded from baselines' publicly available datasets.


https://github.com/cynricfu/MAGNN.

https://github.com/liangchundong/HGCA.


# Dependencies
torch 2.0.0<br>
dgl 1.1.0+cu113<br>
dglgo 0.0.2<br>
networkx 3.0<br>
sklearn 1.2.1<br>
scipy 1.8.1<br>
numpy 1.22.3<br>

# Run
## Link Prediction
```
python main_ACM.py
python main_DBLP.py
python main_YELP.py
```
## Node Classification
```
python nc_ACM.py
python nc_DBLP.py
python nc_YELP.py
```
