from dataset import Dataset
from model import Model
from train import Trainer
from pretrain import PreTrainer
import random


if __name__=='__main__':


    random.seed(901)

    dataset_config = {'feature_file': './data/pubmed/features.txt',
                      'graph_file': './data/pubmed/edges.txt',
                      'walks_rel': './data/pubmed/walks_104010.txt',
                      'label_file': './data/pubmed/group.txt',
                      'walks_att': './data/pubmed/walks_6108.txt',
                      'similar_rate': 3}
    graph = Dataset(dataset_config)


    config = {
        'emb': './emb/pubmed.npy',
        
        'rel_shape': [500, 100],
        'att_shape': [200, 100],
        'rel_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': './Log/pubmed/pretrain_params.pkl',
        
        'drop_prob': 0.2,
        'learning_rate': 1e-4,
        'batch_size': 50,
        'num_epochs': 100,
        'beta': 1,
        'alpha': 0.005,        
        'model_path': './Log/pubmed/pubmed_model.pkl',
    }

    pretrain = PreTrainer(config)
    pretrain.pretrain(graph.R, 'rel')
    pretrain.pretrain(graph.Z, 'att')
    

    model = Model(config)
    train = Trainer(model, config)
    train.train(graph)
    train.infer(graph)

