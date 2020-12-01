# -*- coding: utf-8 -*-

'''
Author: Zhaoning Li
'''

import keras
import numpy as np
import os
import random as rn
import tensorflow as tf
from keras import backend as K
import pickle
from keras.utils import to_categorical
import h5py
from keras.layers import *
import math
from MHSA import MultiHeadSelfAttention
from ChainCRF import ChainCRF
from keras.models import Model
from keras import optimizers
from keras.callbacks import*
from tag2triplet import*
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse


MAX_WLEN = 58
MAX_CLEN = 23
VOCAB_SIZE = 15539
CHAR_SIZE = 69
EXTVEC_DIM = 300
FLAIR_DIM = 1024
CHAR_DIM = 30
NUM_CHAR_CNN_FILTER = 30
CHAR_CNN_KERNEL_SIZE = 3
CHAR_LSTM_SIZE = 25
NUM_ID_CNN_FILTER = 300
ID_CNN_KERNEL_SIZE = 3
DILATION_RATE = (1, 2, 4, 1)
NUM_CLASS = 7


class MaskConv1D(Conv1D):
    def __init__(self, **kwargs):
        super(MaskConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskConv1D, self).call(inputs)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, x, x_flair, x_char, y, batch_size, classifier, pred=False):
        self.list_IDs = list_IDs
        self.x = x
        self.x_flair = x_flair
        self.x_char = x_char
        self.y = y
        self.batch_size = batch_size
        self.classifier = classifier
        self.pred = pred

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index *
                                      self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(list_IDs_temp)

    def __data_generation(self, list_IDs_temp):
        x = []
        if self.pred:
            maxlen = MAX_WLEN
        else:
            maxlen = max([len(np.where(self.x[ID] != 0)[0])
                          for ID in list_IDs_temp])
        x.append(np.zeros((self.batch_size, maxlen)))

        for i, ID in enumerate(list_IDs_temp):
            x[0][i] = self.x[ID][:maxlen]

        if self.x_flair != None:
            x_flair = np.zeros((self.batch_size, maxlen, FLAIR_DIM))
            for i, ID in enumerate(list_IDs_temp):
                x_flair[i] = self.x_flair[ID][:maxlen]
            x.append(x_flair)

        if self.x_char != None:
            if self.pred:
                maxlen_c = MAX_CLEN
            else:
                maxlen_c = max([len(np.where(self.x[ID][_] != 0)[0]) for _ in range(maxlen) for ID in list_IDs_temp])
            x_char = np.zeros((self.batch_size, maxlen, maxlen_c))
            for i, ID in enumerate(list_IDs_temp):
                x_char[i] = self.x_char[ID][:maxlen][:, :maxlen_c]
            x.append(x_char)

        if self.pred:
            return x

        y = np.zeros((self.batch_size, maxlen, 1))
        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.y[ID][:maxlen]

        return x, y


class Data:
    def __init__(self, args):
        self.word2index, self.index2word = pickle.load(
            open(args.file_path+'index/index_w.pkl', 'rb'))
        self.embedding = np.load(
            open(args.file_path+'embedding/extvec_embedding.npy', 'rb'))

        with h5py.File(args.file_path+'train/train.h5', 'r') as fh:
            self.xTrain = fh['xTrain'][:]
            self.yTrain = fh['yTrain'][:]

    def cross_validation(self):
        """
        Return the data of cross validation
        """
        trainIdx, _ = train_test_split([i for i in range(len(self.xTrain))],
                                       test_size=0.,
                                       random_state=args.seed)
        valIdx = trainIdx[args.k_fold *
                          (len(trainIdx)//10):(args.k_fold+1)*(len(trainIdx)//10)]
        trainIdx = [i for i in trainIdx if i not in valIdx]

        train_flair, train_char, eval_x, eval_y = None, None, self.xTrain[
            valIdx], self.yTrain[valIdx]

        if args.use_flair:
            h5f = h5py.File(args.file_path+'embedding/flair.h5', 'r')
            train_flair = h5f['xTrain_flair']

        if args.char_emb != None:
            h5f = h5py.File(args.file_path+'train/train.h5', 'r')
            train_char = h5f['xTrain_c']

        training_generator = DataGenerator(trainIdx,
                                           x=self.xTrain,
                                           x_flair=train_flair,
                                           x_char=train_char,
                                           y=self.yTrain,
                                           batch_size=args.batch_size,                                                          
                                           classifier=args.classifier)
        validation_generator = DataGenerator(valIdx,
                                             x=self.xTrain,
                                             x_flair=train_flair,
                                             x_char=train_char,
                                             y=self.yTrain,                                                                      
                                             batch_size=args.batch_size,                                                          
                                             classifier=args.classifier)
        predict_generator = DataGenerator(valIdx,
                                          x=self.xTrain,
                                          x_flair=train_flair,
                                          x_char=train_char,
                                          y=self.yTrain,
                                          batch_size=args.batch_size,                                                         
                                          classifier=args.classifier,
                                          pred=True)
        return eval_x, eval_y, training_generator, validation_generator, predict_generator

    def train(self):
        """
        Return the data of train
        """
        train_flair, train_char = None, None

        if args.use_flair:
            h5f = h5py.File(args.file_path+'embedding/flair.h5', 'r')
            train_flair = h5f['xTrain_flair']

        if args.char_emb != None:
            h5f_te = h5py.File(args.file_path+'train/train.h5', 'r')
            train_char = h5f_te['xTrain_c']

        training_generator = DataGenerator([i for i in range(len(self.xTrain))],
                                           x=self.xTrain,
                                           x_flair=train_flair,
                                           x_char=train_char,
                                           y=self.yTrain,
                                           batch_size=args.batch_size,
                                           classifier=args.classifier)
        return training_generator


class Evaluate(Callback):
    def __init__(self,
                 args,
                 data,
                 x,
                 x_g,
                 y_true_idx,
                 ap,
                 save_path=None):
        self.pre = []
        self.rec = []
        self.f1 = []
        self.best_f1 = 0.
        self.x = x
        self.x_g = x_g
        self.y_true_idx = y_true_idx
        self.ap = ap
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if args.objective == 'cv':
            y_pred = np.argmax(self.model.predict_generator(self.x_g), axis=-1)[:len(self.x)]
            y_pred_idx = [final_result(
                y_pred[i], [data.index2word[w] for w in self.x[i] if w != 0]) for i in range(len(y_pred))]
            pp = sum([len(i) for i in y_pred_idx if i != 0])
            tp = 0
            for i in range(len(self.y_true_idx)):
                if self.y_true_idx[i] != 0 and y_pred_idx[i] != 0:
                    for m in self.y_true_idx[i]:
                        y_true_cause = [data.index2word[self.x[i][idx]]
                                        for idx in m[0]]
                        y_true_effect = [data.index2word[self.x[i][idx]]
                                         for idx in m[-1]]
                        for n in y_pred_idx[i]:
                            y_pred_cause = [data.index2word[self.x[i][idx]]
                                            for idx in n[0] if self.x[i][idx] != 0]
                            y_pred_effect = [data.index2word[self.x[i][idx]]
                                             for idx in n[-1] if self.x[i][idx] != 0]
                            if y_true_cause == y_pred_cause and y_true_effect == y_pred_effect:
                                tp += 1

            pre = tp / float(pp) if pp != 0 else 0
            rec = tp / float(self.ap) if self.ap != 0 else 0
            f1 = 2 * pre * rec / float(pre + rec) if (pre + rec) != 0 else 0
            self.pre.append(pre)
            self.rec.append(rec)
            self.f1.append(f1)
            if f1 > self.best_f1:
                self.best_f1 = f1
            print(' - val_precision: %.4f - val_recall: %.4f - val_f1_score: %.4f - best_f1_score: %.4f' %
                  (pre, rec, f1, self.best_f1))
            if epoch + 1 == args.num_epochs:
                isExists = os.path.exists(self.save_path+'/cv/'+str(args.seed))
                if not isExists:
                    os.makedirs(self.save_path+'/cv/'+str(args.seed))
                with open(self.save_path+'/cv/'+str(args.seed)+'/'+str(args.k_fold)+'.pkl', 'wb') as fp:
                    pickle.dump((self.pre, self.rec, self.f1), fp, -1)
        if args.objective == 'train':
            #if epoch + 1 == args.num_epochs:
            if epoch + 1 > 40:
                isExists = os.path.exists(self.save_path+'/test')
                if not isExists:
                    os.makedirs(self.save_path+'/test')
                self.model.save_weights(filepath=self.save_path+'/test/'+str(args.seed) + '_{' +
                                str(epoch + 1) + '}' + '.hdf5')


class CausalityExtractor:
    def __init__(self, args):
        self.reproducibility()
        self.kernel_initializer = keras.initializers.glorot_uniform(
            seed=args.seed)
        self.recurrent_initializer = keras.initializers.Orthogonal(
            seed=args.seed)
        self.lr = args.learning_rate
        self.save_path = 'logs/FLAIR-'+str(args.use_flair)+'_CHAR-'+str(args.char_emb) + \
                         '_'+args.backbone.upper() + \
            '_MHSA-'+str(args.use_att)+'_' + args.classifier.upper()

    def reproducibility(self):
        """
        Ensure that the model can obtain reproducible results
        """
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_devices)
        np.random.seed(args.seed)
        rn.seed(args.seed)
        session_conf = tf.ConfigProto(
            device_count={'CPU': args.cpu_core},
            intra_op_parallelism_threads=args.cpu_core,
            inter_op_parallelism_threads=args.cpu_core,
            gpu_options=tf.GPUOptions(allow_growth=True
                                     #per_process_gpu_memory_fraction=0.09
                                     ),
            allow_soft_placement=True)
        tf.set_random_seed(args.seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    def conv_block(self, x, dilation_rate=1, use_dropout=True, name='1'):
        '''
        Utility function to apply conv.
        '''
        x = MaskConv1D(filters=NUM_ID_CNN_FILTER,
                       kernel_size=ID_CNN_KERNEL_SIZE,
                       padding='same',
                       dilation_rate=dilation_rate,
                       kernel_initializer=self.kernel_initializer,
                       name='CONV-'+name)(x)
        x = Activation('relu', name='RELU-'+name)(x)
        if use_dropout:
            x = Dropout(args.dropout_rate, seed=args.seed,
                        name='DROPOUT-'+name)(x)
        return x

    def slm(self, data):
        """
        Returns Sequence Labeling Model.
        """
        seq = Input(shape=(None,), name='INPUT')
        emb = Embedding(VOCAB_SIZE,
                        EXTVEC_DIM,
                        weights=[data.embedding],
                        mask_zero=True,
                        trainable=False, name='WE')(seq)
        input_node = [seq]

        if args.use_flair:
            flair = Input(shape=(None, FLAIR_DIM), name='FLAIR')
            emb = concatenate([emb, flair], axis=-1, name='EMB_FLAIR')
            input_node.append(flair)

        if args.char_emb != None:
            char_embedding = []
            for _ in range(CHAR_SIZE):
                scale = math.sqrt(3.0 / CHAR_DIM)
                char_embedding.append(
                    np.random.uniform(-scale, scale, CHAR_DIM))
            char_embedding[0] = np.zeros(CHAR_DIM)
            char_embedding = np.asarray(char_embedding)

            char_seq = Input(shape=(None, None), name='CHAR_INPUT')
            char_emb = TimeDistributed(
                Embedding(CHAR_SIZE,
                          CHAR_DIM,
                          weights=[char_embedding],
                          mask_zero=True,
                          trainable=True), name='CHAR_EMB')(char_seq)

            if args.char_emb == 'lstm':
                char_emb = TimeDistributed(Bidirectional(LSTM(CHAR_LSTM_SIZE,
                                                              kernel_initializer=self.kernel_initializer,
                                                              recurrent_initializer=self.recurrent_initializer,
                                                              implementation=2,
                                                              return_sequences=False)), name="CHAR_BiLSTM")(char_emb)
                
            if args.char_emb == 'cnn':
                char_emb = TimeDistributed(MaskConv1D(filters=NUM_CHAR_CNN_FILTER,
                                                      kernel_size=CHAR_CNN_KERNEL_SIZE,
                                                      padding='same',
                                                      kernel_initializer=self.kernel_initializer), name="CHAR_CNN")(char_emb)
                char_emb = TimeDistributed(
                    Lambda(lambda x: K.max(x, axis=1)), name="MAX_POOLING")(char_emb)

            input_node.append(char_seq)
            emb = concatenate([emb, char_emb], axis=-1, name='EMB_CHAR')

        if args.backbone == 'lstm':
            
            dec = Bidirectional(LSTM(args.lstm_size,
                                     kernel_initializer=self.kernel_initializer,
                                     recurrent_initializer=self.recurrent_initializer,
                                     dropout=args.dropout_rate,
                                     recurrent_dropout=args.dropout_rate,
                                     implementation=2,
                                     return_sequences=True),
                                merge_mode='concat', name='BiLSTM-1')(emb)
            
            '''
            enc_bilstm = Bidirectional(LSTM(args.lstm_size,
                                            kernel_initializer=self.kernel_initializer,
                                            recurrent_initializer=self.recurrent_initializer,
                                            dropout=args.dropout_rate,
                                            recurrent_dropout=args.dropout_rate,
                                            implementation=2,
                                            return_sequences=True),
                                       merge_mode='concat', name='BiLSTM-1')(emb)
            dec = Bidirectional(LSTM(args.lstm_size,
                                     kernel_initializer=self.kernel_initializer,
                                     recurrent_initializer=self.recurrent_initializer,
                                     dropout=args.dropout_rate,
                                     recurrent_dropout=args.dropout_rate,
                                     implementation=2,
                                     return_sequences=True),
                                merge_mode='concat', name='BiLSTM-2')(enc_bilstm)
            '''
            if args.use_att:
                mhsa = MultiHeadSelfAttention(
                    head_num=args.nb_head, size_per_head=args.size_per_head, kernel_initializer=self.kernel_initializer, name='MHSA')(dec)
                dec = concatenate(
                    [dec, mhsa], axis=-1, name='CONTEXT')

        if args.backbone == 'cnn':
            conv_1 = self.conv_block(
                emb, dilation_rate=DILATION_RATE[0], name='1')
            conv_2 = self.conv_block(
                conv_1, dilation_rate=DILATION_RATE[1], name='2')
            conv_3 = self.conv_block(
                conv_2, dilation_rate=DILATION_RATE[2], name='3')
            dec = self.conv_block(conv_3, dilation_rate=DILATION_RATE[-1],
                                  use_dropout=False, name='4')

        if args.classifier == 'softmax':
            output = TimeDistributed(Dense(NUM_CLASS, activation='softmax',
                                           kernel_initializer=self.kernel_initializer), name='DENSE')(dec)
            loss_func = 'sparse_categorical_crossentropy'

        if args.classifier == 'crf':
            dense = TimeDistributed(Dense(
                NUM_CLASS, activation=None, kernel_initializer=self.kernel_initializer), name='DENSE')(dec)
            crf = ChainCRF(init=self.kernel_initializer, name='CRF')
            output = crf(dense)
            loss_func = crf.sparse_loss

        optimizer = optimizers.Nadam(lr=self.lr, clipnorm=args.clip_norm)
        model = Model(inputs=input_node, outputs=output)
        model.compile(loss=loss_func, optimizer=optimizer)
        return model

    def cv(self, data):
        """
        Cross validation
        """
        model = self.slm(data)
        eval_x, eval_y, training_generator, validation_generator, predict_generator = data.cross_validation()
        model.summary()

        save_path = self.save_path
        isExists = os.path.exists(save_path)
        if not isExists:
            os.makedirs(save_path)

        y_true = eval_y.reshape(eval_y.shape[0], MAX_WLEN)
        y_true_idx = [final_result(y_true[i], [
            data.index2word[w] for w in eval_x[i] if w != 0]) for i in range(len(y_true))]
        ap = sum([len(i) for i in y_true_idx if i != 0])
        evaluator = Evaluate(args,
                             data,
                             eval_x,
                             predict_generator,
                             y_true_idx,
                             ap,
                             save_path=save_path)
        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, verbose=1, cooldown=5, min_lr=0.00005)
        model.fit_generator(training_generator,
                            epochs=args.num_epochs,
                            verbose=1,
                            validation_data=validation_generator,
                            callbacks=[evaluator, reduce_lr],
                            shuffle=False)

    def train(self, data):
        """
        Train
        """
        model = self.slm(data)
        training_generator = data.train()
        model.summary()

        save_path = self.save_path
        isExists = os.path.exists(save_path)
        if not isExists:
            os.makedirs(save_path)

        evaluator = Evaluate(args, None, None, None, None,
                             None, save_path=save_path)
        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, verbose=1, cooldown=5, min_lr=0.00005)
        model.fit_generator(training_generator,
                            epochs=args.num_epochs,
                            verbose=1,
                            callbacks=[evaluator, reduce_lr],
                            shuffle=False)

        
parser = argparse.ArgumentParser()
parser.add_argument('-fp', '--file_path', type=str, default="your path to /data/", help="")
parser.add_argument('-s', '--seed', type=int, default=0, help="")
parser.add_argument('-kf', '--k_fold', type=int, default=0)
parser.add_argument('-cuda', '--cuda_devices', type=int, default=0)
parser.add_argument('-cpu', '--cpu_core', type=int, default=1)
parser.add_argument('-o', '--objective', type=str, default="train", help="train or cv")
parser.add_argument('-cse', '--use_flair', type=bool, default=True, help="")
parser.add_argument('-chr', '--char_emb', type=str, default=None, help="lstm or cnn or None")
parser.add_argument('-b', '--backbone', type=str, default="lstm", help="lstm or cnn")
parser.add_argument('-ls', '--lstm_size', type=int, default=256, help="")
parser.add_argument('-dp', '--dropout_rate', type=float, default=0.5, help="")
parser.add_argument('-att', '--use_att', type=bool, default=False, help="")
parser.add_argument('-nh', '--nb_head', type=int, default=3, help="")
parser.add_argument('-hs', '--size_per_head', type=int, default=8, help="")
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="")
parser.add_argument('-cl', '--classifier', type=str, default="crf", help="softmax or crf")
parser.add_argument('-cn', '--clip_norm', type=float, default=5.0, help="")
parser.add_argument('-n', '--num_epochs', type=int, default=200, help="")
parser.add_argument('-bs', '--batch_size', type=int, default=16, help="")
args = parser.parse_args()


data = Data(args)
extractor = CausalityExtractor(args)

for s in [6, 66, 666, 6666, 66666]:
    args.seed = s
    if args.objective == 'train':
        extractor.train(data)
    kf_iter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for kf in kf_iter:
        args.k_fold = kf
        if args.objective == 'cv':
            extractor.cv(data)