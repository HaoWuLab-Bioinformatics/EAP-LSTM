import sys
sys.path.append('..')
import tensorflow
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import network
import numpy as np
import pandas as pd
from utils import IOHelper,SequenceHelper
from scipy.stats import spearmanr
from keras.optimizers import Adam,RMSprop,Adagrad,Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy import stats
from sklearn.metrics import mean_squared_error
import random
import keras
from tensorflow.keras import backend as K
random.seed(2024)
import math



def summary_statistics(X, Y, set, model):
    pred = model.predict(X)
    print(type(pred))
    mse=mean_squared_error(Y, pred.squeeze())
    pcc=stats.pearsonr(Y, pred.squeeze())[0]
    scc=stats.spearmanr(Y, pred.squeeze())[0]
    print(' MSE = ' + str("{0:0.4f}".format(mse)))
    print(' PCC = ' + str("{0:0.4f}".format(pcc)))
    print(' SCC = ' + str("{0:0.4f}".format(scc)))
    if set=='test':
        with open('../results/result_human.txt', 'a') as f:
            f.write('cell_line '+ str(cell_line) +'\n'  
                    'result ' + ' mse ' + str(mse) + ' pcc ' + str(pcc) + ' scc ' + str(scc)+ '\n')

def prepare_input(set):
    # Convert sequences to one-hot encoding matrix
    file_seq = str("Sequences_" + set + ".fa")


    input_fasta_data_A = IOHelper.get_fastas_from_file(file_seq, uppercase=True)    #返回的是dataframe格式

    # get length of first sequence
    sequence_length = len(input_fasta_data_A.sequence.iloc[0]) #iloc（0） 返回第0行的值

    # Convert sequence to one hot encoding matrix   #one hot 编码
    seq_matrix_A = SequenceHelper.do_one_hot_encoding(input_fasta_data_A.sequence, sequence_length,
                                                      SequenceHelper.parse_alpha_to_seq)
    print(seq_matrix_A.shape)

    X = np.nan_to_num(seq_matrix_A)  # Replace NaN with zero and infinity with large finite numbers
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    Activity = pd.read_table("Sequences_activity_" + set + ".txt")
    Y_dev = Activity.Dev_log2_enrichment
    Y_hk = Activity.Hk_log2_enrichment
    Y = [Y_dev, Y_hk]

    print(set)

    return input_fasta_data_A.sequence, seq_matrix_A, X_reshaped, Y
def word_embedding(filename, index, word2vec):
    f = open(filename, 'r')
    sequence = []
    for line in f.readlines():
        if line[0] != '>':
            seq = line.upper().strip('\n')
            sequence.append(seq)


    k = 3
    kmer_list = []
    for number in range(len(sequence)):
        seq = []
        for i in range(len(sequence[number]) - k + 1):
            if sequence[number][i:i + k] in index:
                ind = index.index(sequence[number][i:i + k])
                seq.append(ind)
            else: seq.append(64)
        kmer_list.append(seq)

    '''sum_length = 0
    cnt = 0
    for number in range(len(sequence)):
        sum_length += (len(sequence[number]) - k + 1)
        cnt = number
    average_length = round(sum_length / (cnt + 1))'''

    feature_word2vec = []
    for number in range(len(kmer_list)):
        # print(number)
        feature_seq = []
        for i in range(len(kmer_list[number])):
            feature_seq_num=[]
            kmer_index = kmer_list[number][i]
            for j in word2vec[kmer_index].tolist():
                feature_seq_num.append(j)
            feature_seq.append(feature_seq_num)

        feature_seq_tensor_avg = torch.Tensor(feature_seq)
        print(feature_seq_tensor_avg.shape)
        feature_seq_numpy = feature_seq_tensor_avg.numpy()
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = np.squeeze(feature_seq_numpy)
        feature_seq_numpy = feature_seq_numpy.astype('float64')
        feature_seq_list = feature_seq_numpy.tolist()

        feature_word2vec.append(feature_seq_list)

    return feature_word2vec
def FeatureConcet(cell_line,set):
    features = ['mismatch', 'RCkmer']
    for feature in features:
        feature_path = 'feature_human/'+ cell_line +'/'+ set +'/'+feature + '.csv'
        fea = np.loadtxt(feature_path, delimiter=',')[:,1:]
        if feature == 'mismatch':
            x = fea
        else:
            x = np.concatenate((x, fea), axis=1)
    return x
def Get_STARR_signal(cell):
    TF_signal_pos = pd.read_table("data_human/"+cell + "_STARR_pos.fasta", sep=' ', header=None)
    TF_signal_neg = pd.read_table("data_human/"+cell + "_STARR_neg.fasta", sep=' ', header=None)
    TF_signal = pd.concat([TF_signal_pos,TF_signal_neg]).iloc[:,3:].fillna(0) #nan 变为0
    TF_signal = np.array(TF_signal,dtype="float32")
    return TF_signal

def Get_Signal_3003_1(cell,set):
    DNase = pd.read_table("data_human/"+cell +"/" + cell+"_"+set+"_DNase.txt",sep=' ', header=None).iloc[:, 3:]
    H3K27ac = pd.read_table("data_human/"+cell +"/" + cell+"_"+set+"_H3K27ac.txt",sep=' ', header=None).iloc[:, 3:]
    H3K4me1 = pd.read_table("data_human/"+cell +"/" + cell+"_"+set+"_H3K4me1.txt",sep=' ', header=None).iloc[:, 3:]
    Signal = pd.concat([DNase,H3K27ac,H3K4me1],axis=1).fillna(0)
    Signal = np.array(Signal, dtype="float32")
    Signal = np.expand_dims(Signal,2)
    return Signal
def process(cell_line,set):
    f = open('feature_human/word2vec/index_file.txt', 'r')
    index = f.read()
    f.close()
    index = index.strip().split('\n')
    word2vec = np.loadtxt('feature_human/word2vec/vectors_file.txt')
    filename = 'data_human/'+cell_line + "/"+cell_line+"_"+ set+".fasta"
    feature_word2vec = word_embedding(filename, index, word2vec)
    feature_word2vec = np.array(feature_word2vec)
    print(feature_word2vec.shape)
    print(feature_word2vec.dtype)
    feature_word2vec = feature_word2vec.astype('float64')
    return feature_word2vec

for cell_line in ['A549','HCT116','HepG2','K562','MCF-7']:
    X_tarin_signal = Get_Signal_3003_1(cell_line,'train')
    X_test_signal = Get_Signal_3003_1(cell_line,'test')
    X_valid_signal = Get_Signal_3003_1(cell_line,'valid')
    X_tarin_signal = np.log2(X_tarin_signal + 1)
    X_test_signal = np.log2(X_test_signal + 1)
    X_valid_signal = np.log2(X_valid_signal + 1)

    X_train_1 = np.load('feature_human/'+ cell_line +'/train/word2vec_1.pt.npy')
    X_train_2 = np.load('feature_human/'+ cell_line +'/train/word2vec_2.pt.npy')
    X_train = np.concatenate([X_train_1,X_train_2],axis=0)
    X_test = np.load('feature_human/'+ cell_line +'/test/word2vec.pt.npy')
    X_valid = np.load('feature_human/'+ cell_line +'/valid/word2vec.pt.npy')


    X_train_f = FeatureConcet(cell_line,'train')
    X_test_f = FeatureConcet(cell_line,'test')
    X_valid_f = FeatureConcet(cell_line,'valid')
    X_train_f = np.expand_dims(X_train_f,2)
    X_test_f = np.expand_dims(X_test_f,2)
    X_valid_f = np.expand_dims(X_valid_f,2)


    Y_train = np.loadtxt("data_human/" + cell_line + "/" + cell_line+"_train_labels_2.txt")
    Y_valid = np.loadtxt("data_human/" + cell_line + "/" + cell_line+"_valid_labels_2.txt")
    Y_test = np.loadtxt("data_human/" + cell_line + "/" + cell_line+"_test_labels_2.txt")
    Y_train = np.log2(Y_train + 1)
    Y_valid = np.log2(Y_valid + 1)
    Y_test = np.log2(Y_test + 1)


    print(X_train.shape)
    print(X_train_f.shape)
    print(X_tarin_signal.shape)
    print(Y_train.shape)
    print(X_valid.shape)
    print(X_valid_f.shape)
    print(X_valid_signal.shape)
    print(Y_valid.shape)
    print(X_test.shape)
    print(X_test_f.shape)
    print(X_test_signal.shape)
    print(Y_test.shape)

    data_shape1 = X_train.shape[1:3]
    data_shape2 = X_train_f.shape[1:3]
    data_shape3 = X_tarin_signal.shape[1:3]

    learing_rate = 0.002
    BATCH_SIZE = 128
    kernel_num = 64
    kernel_size1 = 9
    kernel_size2= 3
    kernel_size3= 9
    kernel_size4= 5
    dropout_rate= 0.2
    pool_size = 3
    MAX_EPOCH = 200
    stride = 1

    initial_lr = 0.1
    min_lr = 0.001
    one_epoch_batchs = int(X_train.shape[0]/BATCH_SIZE)
    print(one_epoch_batchs)
    total_epochs = 50
    total_step = one_epoch_batchs * 3
    warmup_step = int(total_step * 0.25)
    print_step = one_epoch_batchs





    class CosineWarmupDecay(tensorflow.keras.optimizers.schedules.LearningRateSchedule):
        '''
        initial_lr: 初始的学习率
        min_lr: 学习率的最小值
        max_lr: 学习率的最大值
        warmup_step: 线性上升部分需要的step
        total_step: 第一个余弦退火周期需要对总step
        multi: 下个周期相比于上个周期调整的倍率
        print_step: 多少个step并打印一次学习率
        '''
        def __init__(self, initial_lr, min_lr, warmup_step, total_step, multi, print_step):
            super(CosineWarmupDecay, self).__init__()
            self.initial_lr = tensorflow.cast(initial_lr, dtype=tensorflow.float32)
            self.min_lr = tensorflow.cast(min_lr, dtype=tensorflow.float32)
            self.warmup_step = warmup_step
            self.total_step = total_step
            self.multi = multi
            self.print_step = print_step
            self.learning_rate_list = []
            self.step = 0
        def __call__(self,step):
            if  self.step>=self.total_step:
                self.warmup_step = self.warmup_step * (1 + self.multi)
                self.total_step = self.total_step * (1 + self.multi)
                self.step = 0
            decayed_learning_rate = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                                    (1 + tensorflow.math.cos(math.pi * (self.step-self.warmup_step)/(self.total_step-self.warmup_step)))
            k = (self.initial_lr - self.min_lr) / self.warmup_step
            warmup = k * self.step + self.min_lr
            decayed_learning_rate = tensorflow.where(self.step<self.warmup_step, warmup, decayed_learning_rate)
            if step % self.print_step == 0:
                print('learning_rate has changed to: ', decayed_learning_rate.numpy().item())
            self.learning_rate_list.append(decayed_learning_rate.numpy().item())
            self.step = self.step + 1
            print(self.step)
            return decayed_learning_rate


    def Spearman(y_true, y_pred):
        return (tensorflow.py_function(spearmanr, [tensorflow.cast(y_pred, tensorflow.float32),
                                           tensorflow.cast(y_true, tensorflow.float32)], Tout=tensorflow.float32))
    def pearsonr(y_true, y_pred):
        return (tensorflow.py_function(spearmanr, [tensorflow.cast(y_pred, tensorflow.float32),
                                           tensorflow.cast(y_true, tensorflow.float32)], Tout=tensorflow.float32))

    lr_schedule = CosineWarmupDecay (initial_lr=initial_lr,
                                      min_lr=min_lr,
                                      warmup_step=warmup_step,
                                      total_step=total_step,
                                      multi=0.25,
                                      print_step=print_step)

    filepath = "model/human/"+cell_line+"_model.hdf5"
    kf = KFold(n_splits=10,shuffle=True, random_state=10)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    model = network.EAP_LSTM_human(data_shape1,data_shape2,data_shape3,kernel_size1,kernel_size2,kernel_size3,kernel_size4,pool_size,dropout_rate, stride)
    model.compile(loss='mse',optimizer=Adam(lr=lr_schedule(step=0)), metrics=[Spearman])
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
    model.fit([X_train,X_train_f,X_tarin_signal],Y_train.max(1),batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
                 validation_data=([X_valid,X_valid_f,X_valid_signal],Y_valid.max(1)),
                 callbacks=[checkpoint,early_stopping_monitor])
    summary_statistics([X_train, X_train_f,X_tarin_signal], Y_train.max(1), 'train', model)
    summary_statistics([X_valid, X_valid_f,X_valid_signal], Y_valid.max(1), 'valid', model)
    summary_statistics([X_test, X_test_f,X_test_signal], Y_test.max(1), 'test',model)
