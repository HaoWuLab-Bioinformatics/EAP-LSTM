import tensorflow
import torch
from sklearn.model_selection import train_test_split

import network
import numpy as np
import pandas as pd
from utils import IOHelper,SequenceHelper
from scipy.stats import spearmanr
from keras.optimizers import Adam,RMSprop,Adagrad,Adadelta
from keras.callbacks import EarlyStopping
from scipy import stats
from sklearn.metrics import mean_squared_error
import random
random.seed(2024)
def summary_statistics(X, Y, set, task):
    pred = model.predict(X)
    if task == "Dev":
        i = 0
    if task == "Hk":
        i = 1
    mse=mean_squared_error(Y, pred[i].squeeze())
    pcc=stats.pearsonr(Y, pred[i].squeeze())[0]
    scc=stats.spearmanr(Y, pred[i].squeeze())[0]
    print(set + ' MSE ' + task + ' = ' + str("{0:0.4f}".format(mean_squared_error(Y, pred[i].squeeze()))))
    print(set + ' PCC ' + task + ' = ' + str("{0:0.4f}".format(stats.pearsonr(Y, pred[i].squeeze())[0])))
    print(set + ' SCC ' + task + ' = ' + str("{0:0.4f}".format(stats.spearmanr(Y, pred[i].squeeze())[0])))
    if set == 'test':
        with open('results/result_drosophila.txt', 'a') as f:
            f.write('cell_line drosophila'   +
                    ' X_train ' + str(X_train.shape) +
                    ' X_valid ' + str(X_valid.shape) +
                    ' X_test ' + str(X_test.shape) +
                    ' task ' + str(task)+'\n'
                     'result ' + ' mse ' + str(mse) + ' pcc ' + str(pcc) + ' scc ' + str(
                scc) + '\n')


def prepare_input(set):
    # Convert sequences to one-hot encoding matrix
    file_seq = str("data_drosophila/Sequences_" + set + ".fa")


    input_fasta_data_A = IOHelper.get_fastas_from_file(file_seq, uppercase=True)    #返回的是dataframe格式

    # get length of first sequence
    sequence_length = len(input_fasta_data_A.sequence.iloc[0]) #iloc（0） 返回第0行的值

    # Convert sequence to one hot encoding matrix   #one hot 编码
    seq_matrix_A = SequenceHelper.do_one_hot_encoding(input_fasta_data_A.sequence, sequence_length,
                                                      SequenceHelper.parse_alpha_to_seq)
    print(seq_matrix_A.shape)

    X = np.nan_to_num(seq_matrix_A)  # Replace NaN with zero and infinity with large finite numbers
    X_reshaped = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    Activity = pd.read_table("data_drosophila/Sequences_activity_" + set + ".txt")
    Y_dev = Activity.Dev_log2_enrichment
    Y_hk = Activity.Hk_log2_enrichment
    Y = [Y_dev, Y_hk]

    print(set)

    return input_fasta_data_A.sequence, seq_matrix_A, X_reshaped, Y
def word_embedding(filename, index, word2vec):
    f = open(filename, 'r')
    sequence = []
    for line in f.readlines():
        if line[0] != ' ':
            if line[0] != '>':
                sequence.append(line.upper().strip('\n'))

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

def process(set):
    f = open('feature_drosophila/word2vec/index_file.txt', 'r')
    index = f.read()
    f.close()
    index = index.strip().split('\n')
    word2vec = np.loadtxt('feature_drosophila/word2vec/vectors_file.txt')
    filename = "Sequences_" + set + ".fa"
    feature_word2vec = word_embedding(filename, index, word2vec)
    feature_word2vec = np.array(feature_word2vec)
    print(feature_word2vec.shape)
    print(feature_word2vec.dtype)
    feature_word2vec = feature_word2vec.astype('float32')
    return feature_word2vec

def FeatureConcet(set):
    features = ['mismatch', 'RCkmer']
    for feature in features:
        feature_path = 'feature_drosophila/'+set+'/'+ feature + '.csv'
        fea = np.loadtxt(feature_path, delimiter=',')[:,1:]

        if feature == 'mismatch':
            x = fea
        else:
            x = np.concatenate((x, fea), axis=1)
    return x



X_train = process('Train')
X_test = process('Test')
X_valid = process('Val')

X_train_f = FeatureConcet('train')
X_test_f = FeatureConcet('test')
X_valid_f = FeatureConcet('val')

X_train_f = np.expand_dims(X_train_f, 2)
X_test_f = np.expand_dims(X_test_f, 2)
X_valid_f = np.expand_dims(X_valid_f, 2)

Activity = pd.read_table("data_drosophila/Sequences_activity_Train" + ".txt")
Y_dev = Activity.Dev_log2_enrichment
Y_hk = Activity.Hk_log2_enrichment
Y_train = [Y_dev, Y_hk]

Activity = pd.read_table("data_drosophila/Sequences_activity_Val" + ".txt")
Y_dev = Activity.Dev_log2_enrichment
Y_hk = Activity.Hk_log2_enrichment
Y_valid = [Y_dev, Y_hk]

Activity = pd.read_table("data_drosophila/Sequences_activity_Test" + ".txt")
Y_dev = Activity.Dev_log2_enrichment
Y_hk = Activity.Hk_log2_enrichment
Y_test = [Y_dev, Y_hk]

Y_train = np.array(Y_train).T
Y_valid = np.array(Y_valid).T
Y_test = np.array(Y_test).T
# _, X_train,_,X_train_f,_,Y_train = train_test_split(X_train,X_train_f, Y_train, test_size=bili, random_state=42)
# _, X_valid,_,X_valid_f,_, Y_valid = train_test_split(X_valid, X_valid_f,Y_valid, test_size=bili, random_state=42)
Y_train =Y_train.T.tolist()
Y_valid = Y_valid.T.tolist()
Y_test = Y_test.T.tolist()

data_shape1 = X_train.shape[1:3]
data_shape2 = X_train_f.shape[1:3]
print(X_train.shape)
print(X_train_f.shape)
print(X_valid.shape)
print(X_test.shape)


learning_rate= 0.002
kernel_num = 64
kernel_size1 = 9
kernel_size2= 3
kernel_size3= 9
kernel_size4= 5
dropout_rate= 0.2
pool_size = 3
MAX_EPOCH = 200
BATCH_SIZE = 128
stride = 1
def Spearman(y_true, y_pred):
    return (tensorflow.py_function(spearmanr, [tensorflow.cast(y_pred, tensorflow.float32),
                                       tensorflow.cast(y_true, tensorflow.float32)], Tout=tensorflow.float32))


def Vae_Loss():
    def reconstruction_loss(y_true, y_pred):
        recon_loss = tensorflow.keras.losses.mean_squared_error(y_true, y_pred)
        return recon_loss
    return reconstruction_loss

filepath = "model/drosophila_model.hdf5"



model = network.EAP_LSTM_drosophila(data_shape1,data_shape2,kernel_size1,kernel_size2,kernel_size3,kernel_size4,pool_size,dropout_rate, stride)
model.compile(loss=['mse', 'mse'],loss_weights=[1, 1],optimizer=Adam(lr=learning_rate), metrics=[Spearman])
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)
model.fit([X_train,X_train_f],Y_train,batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
             validation_data=([X_valid,X_valid_f],Y_valid),
             callbacks=[early_stopping_monitor])
print(X_train.shape)
summary_statistics([X_train,X_train_f], Y_train[0], "train", "Dev")
summary_statistics([X_train,X_train_f], Y_train[1], "train", "Hk")
summary_statistics([X_valid,X_valid_f], Y_valid[0], "validation", "Dev")
summary_statistics([X_valid,X_valid_f], Y_valid[1], "validation", "Hk")
summary_statistics([X_test,X_test_f], Y_test[0], "test", "Dev")
summary_statistics([X_test,X_test_f], Y_test[1], "test", "Hk")

