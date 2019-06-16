import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from evaluate import *


class Trainer(object):

    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.rel_input_dim = config['rel_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.rel_shape = config['rel_shape']
        self.att_shape = config['att_shape']
        self.drop_prob = config['drop_prob']
        
        
        self.beta = config['beta']
        self.alpha = config['alpha']
        
        
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']


        self.r = tf.placeholder(tf.float32, [None, self.rel_input_dim])
        self.z = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.x = tf.placeholder(tf.float32, [None, None])

        self.neg_r = tf.placeholder(tf.float32, [None, self.rel_input_dim])
        self.neg_z = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.neg_x = tf.placeholder(tf.float32, [None, None])

        self.optimizer, self.loss = self._build_training_graph()
        self.rel_E, self.att_E, self.H = self._build_eval_graph()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_training_graph(self):
        rel_E, rel_recon = self.model.forward_rel(self.r, drop_prob=self.drop_prob, reuse=False)
        neg_rel_E, neg_rel_recon = self.model.forward_rel(self.neg_r, drop_prob=self.drop_prob, reuse=True)

        att_E, att_recon = self.model.forward_att(self.z, drop_prob=self.drop_prob, reuse=False)
        neg_att_E, neg_att_recon = self.model.forward_att(self.neg_z, drop_prob=self.drop_prob, reuse=True)

        
        recon_loss_r = tf.reduce_mean(tf.reduce_sum(tf.square(self.r - rel_recon), 1))
        recon_loss_z = tf.reduce_mean(tf.reduce_sum(tf.square(self.z - att_recon), 1))
        recon_loss = recon_loss_r  + recon_loss_z 


        #===============cross modality proximity==================
        pre_logit_pos = tf.reduce_sum(tf.multiply(rel_E, att_E), 1)
        pre_logit_neg_1 = tf.reduce_sum(tf.multiply(neg_rel_E, att_E), 1)
        pre_logit_neg_2 = tf.reduce_sum(tf.multiply(rel_E, neg_att_E), 1)

        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pre_logit_pos), logits=pre_logit_pos)
        neg_loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_1), logits=pre_logit_neg_1)
        neg_loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_2), logits=pre_logit_neg_2)
        cross_modal_loss = tf.reduce_mean(pos_loss + neg_loss_1 + neg_loss_2)

        
        loss = recon_loss * self.beta + cross_modal_loss * self.alpha


        vars_rel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'rel_encoder')
        vars_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'att_encoder')
        print(vars_rel)

        
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_rel+vars_att)

        return opt, loss

    def _build_eval_graph(self):
        rel_E, _ = self.model.forward_rel(self.r, drop_prob=0.0, reuse=True)
        att_E, _ = self.model.forward_att(self.z, drop_prob=0.0, reuse=True)
        H = tf.concat([tf.nn.l2_normalize(rel_E, dim=1), tf.nn.l2_normalize(att_E, dim=1)], axis=1)

        return rel_E, att_E, H



    def train(self, graph):

        for epoch in range(self.num_epochs):

            idx1, idx2 = self.generate_samples(graph)

            index = 0
            cost = 0.0
            cnt = 0
            while True:
                if index > graph.num_nodes:
                    break
                if index + self.batch_size < graph.num_nodes:
                    mini_batch1 = graph.sample_by_idx(idx1[index:index + self.batch_size])
                    mini_batch2 = graph.sample_by_idx(idx2[index:index + self.batch_size])
                else:
                    mini_batch1 = graph.sample_by_idx(idx1[index:])
                    mini_batch2 = graph.sample_by_idx(idx2[index:])
                index += self.batch_size

                loss, _ = self.sess.run([self.loss, self.optimizer],
                                        feed_dict={self.r: mini_batch1.R,
                                                   self.z: mini_batch1.Z,
                                                   self.neg_r: mini_batch2.R,
                                                   self.neg_z: mini_batch2.Z,
                                                   self.x: mini_batch1.X,
                                                   self.neg_x: mini_batch2.X})

                cost += loss
                cnt += 1

                if graph.is_epoch_end:
                    break
            cost /= cnt

            if epoch % 10 == 0:

                train_emb = None
                train_label = None
                while True:
                    mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

                    emb = self.sess.run(self.H,
                                        feed_dict={self.r: mini_batch.R,
                                                   self.z: mini_batch.Z})
                    if train_emb is None:
                        train_emb = emb
                        train_label = mini_batch.Y
                    else:
                        train_emb = np.vstack((train_emb, emb))
                        train_label = np.vstack((train_label, mini_batch.Y))

                    if graph.is_epoch_end:
                        break
                micro_f1, macro_f1 = check_multi_label_classification(train_emb, train_label, 0.5)
                print('Epoch-{}, loss: {:.4f}, Micro_f1 {:.4f}, Macro_fa {:.4f}'.format(epoch, cost, micro_f1, macro_f1))
        self.save_model()


    def infer(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)
            emb = self.sess.run(self.H, feed_dict={self.r: mini_batch.R,
                                                   self.z: mini_batch.Z})

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb))
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break


        test_ratio = np.arange(0.5, 1.0, 0.2)
        dane = []
        
        np.save(self.config['emb'],train_emb)
        for tr in range(1,10,1):
            tr = tr/10
            print('============train ration-{}=========='.format(1 - tr))
            micro, macro = multi_label_classification(train_emb, train_label, tr)
            dane.append('{:.4f}'.format(micro) + ' & ' + '{:.4f}'.format(macro))
        print(' & '.join(dane))



    def generate_samples(self, graph):        
        R = []
        Z = []

        order = np.arange(graph.num_nodes)
        np.random.shuffle(order)

        index = 0
        while True:
            if index > graph.num_nodes:
                break
            if index + self.batch_size < graph.num_nodes:
                mini_batch = graph.sample_by_idx(order[index:index + self.batch_size])
            else:
                mini_batch = graph.sample_by_idx(order[index:])
            index += self.batch_size

            rel_E, att_E = self.sess.run([self.rel_E, self.att_E],
                                         feed_dict={self.r: mini_batch.R,
                                                    self.z: mini_batch.Z})
            R.extend(rel_E)
            Z.extend(att_E)

        R = np.array(R)
        Z = np.array(Z)

        R = preprocessing.normalize(R, norm='l2')
        Z = preprocessing.normalize(Z, norm='l2')

        sim = np.dot(R, Z.T)
        neg_idx = np.argmin(sim, axis=1)


        return order, neg_idx


    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
