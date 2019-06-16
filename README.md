# FATNet

# This is a TensorFlow implementation of of Fusing Attributed and Topological Global-Relations for Network Embedding.

## Requirements
* tensorflow (>0.14)

## Preprocess data

Run the walk.py for data preprocessing.

## Run experiments

```
python train_cora.py
```
## Data
In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).
* walk file(you can use walks.py to generate).

## Tips
Please create the following log folders in this project directory.
```
./Log/cora
./Log/citeseer
./Log/wiki
./Log/pubmed
./Log/blogcatalog
```
