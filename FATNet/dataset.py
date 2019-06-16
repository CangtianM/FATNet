import numpy as np
import linecache

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.feature_file = config['feature_file']
        self.label_file = config['label_file']
        self.walks_rel = config['walks_rel']
        self.walks_att = config['walks_att']
        self.similar_rate = config['similar_rate']

        self.W, self.X, self.R, self.Z, self.Y, self.walks_att = self._load_data()

        self.num_nodes = self.W.shape[0]
        self.num_feas = self.Z.shape[1]
        self.num_classes = self.Y.shape[1]
        self.num_edges = np.sum(self.W) / 2
        print('nodes {}, edes {}, features {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_feas, self.num_classes))


        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False

    def _load_data(self):
        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.zeros((num_nodes, num_classes), dtype=np.int32)
        for idx, y in enumerate(Y):
            L[idx][y] = 1
            
        #==========load graph========
        W = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            W[idx2, idx1] = 1.0
            W[idx1, idx2] = 1.0
            
        #=========load walks========
        
        X = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.walks_rel)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            X[idx2, idx1] = 1.0
            X[idx1, idx2] = 1.0
            
        walks_att = np.zeros([num_nodes,num_nodes],dtype=np.float32)
        lines = linecache.getlines(self.walks_att)
        lines = [line.rstrip('\n') for line in lines]
        
        for line in lines:
            line = line.split(' ')
            walks_att[int(line[0]),int(line[1])] = 1
            walks_att[int(line[1]),int(line[0])] = 1     
            

        #=========load feature==========
        lines = linecache.getlines(self.feature_file)
        lines = [line.rstrip('\n') for line in lines]

        num_features = len(lines[0].split(' ')) - 1
        Z = np.zeros((num_nodes, num_features*2), dtype=np.float32)
        Z_node = np.zeros((num_nodes, num_features), dtype=np.float32)
        Z_neighbor = np.zeros((num_nodes, num_features), dtype=np.float32)
        
        for line in lines:
            line = line.split(' ')
            node_id = node_map[line[0]]
            Z_node[node_id] = np.array([float(x) for x in line[1:]])
        for i in range(num_nodes):
            neighbor_index         = np.ravel(np.argwhere(walks_att[i,:]==1))
            for _ in neighbor_index:
                Z_neighbor[i] += Z_node[_]
            if len(neighbor_index)!=0:
                Z_neighbor[i] = Z_neighbor[i] / len(neighbor_index)   

        Z = np.hstack((Z_node,Z_neighbor))
        
        R = np.zeros((num_nodes, num_nodes))
        R = X
        #=========attribute similar==========
        simi= np.dot(Z_node, Z_node.T)        
        average = sum(sum(simi))/(Z_node.shape[0]*Z_node.shape[0])
        for i in range(num_nodes):
            for j in range(num_nodes):
                if simi[i,j] > average*self.similar_rate:
                    R[i,j] = 1
                
        
        return W, X, R, Z, L, walks_att


    def sample(self, batch_size, do_shuffle=True, with_label=True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self._order)
            else:
                self._order = np.sort(self._order)
            self.is_epoch_end = False
            self._index_in_epoch = 0

        mini_batch = Dotdict()
        end_index = min(self.num_nodes, self._index_in_epoch + batch_size)
        cur_index = self._order[self._index_in_epoch:end_index]
        mini_batch.R = self.R[cur_index]
        mini_batch.adj = self.W[cur_index][:, cur_index]
        mini_batch.X = self.X[cur_index][:, cur_index]
        mini_batch.Z = self.Z[cur_index]
        if with_label:
            mini_batch.Y = self.Y[cur_index]

        if end_index == self.num_nodes:
            end_index = 0
            self.is_epoch_end = True
        self._index_in_epoch = end_index

        return mini_batch

    def sample_by_idx(self, idx):
        mini_batch = Dotdict()
        mini_batch.R = self.R[idx]
        mini_batch.Z = self.Z[idx]
        mini_batch.X = self.X[idx][:, idx]

        return mini_batch

