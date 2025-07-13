import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset,RNASSDataGenerator_input
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections
from ppa.ppa_framework import SecondaryStructurePredictor
from collections import defaultdict

import subprocess
args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess





def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file


def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in torch.nonzero(seq_embedding,as_tuple=False)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict





def creatmat(data, device=None):
    if device==None:
        # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device =torch.device('cpu')

    with torch.no_grad():
        data = ''.join(['AUCG'[list(d).index(1)] if 1 in d else 'N' for d in data ])   # 
        paired = defaultdict(int, {'AU':2, 'UA':2, 'GC':3, 'CG':3, 'UG':0.8, 'GU':0.8})

        mat = torch.tensor([[paired[x+y] for y in data] for x in data]).to(device)
        n = len(data)

        i, j = torch.meshgrid(torch.arange(n).to(device), torch.arange(n).to(device), indexing=None)
        t = torch.arange(30).to(device)
        m1 = torch.where((i[:, :, None] - t >= 0) & (j[:, :, None] + t < n), mat[torch.clamp(i[:,:,None]-t, 0, n-1), torch.clamp(j[:,:,None]+t, 0, n-1)], 0)
        m1 = m1.float()
        m1 *= torch.exp(-0.5*t*t)

        m1_0pad = torch.nn.functional.pad(m1, (0, 1))
        first0 = torch.argmax((m1_0pad==0).to(int), dim=2)
        to0indices = t[None,None,:]>first0[:,:,None]
        m1[to0indices] = 0
        m1 = m1.sum(dim=2)

        t = torch.arange(1, 30).to(device)
        m2 = torch.where((i[:, :, None] + t < n) & (j[:, :, None] - t >= 0), mat[torch.clamp(i[:,:,None]+t, 0, n-1), torch.clamp(j[:,:,None]-t, 0, n-1)], 0)
        m2=m2.float()
        m2 *= torch.exp(-0.5*t*t)

        m2_0pad = torch.nn.functional.pad(m2, (0, 1))
        first0 = torch.argmax((m2_0pad==0).to(int), dim=2)
        to0indices = torch.arange(29).to(device)[None,None,:]>first0[:,:,None]
        m2[to0indices] = 0
        m2 = m2.sum(dim=2)
        m2[m1==0] = 0

        return (m1+m2).to(torch.device('cpu'))
def get_cut_len(data_len,set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l

from itertools import product
perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1,3],[3,1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]
def get_input_data(sequence):

    data_seq=sequence
    data_len=len(sequence)

    l = get_cut_len(data_len,80)
    data_fcn = np.zeros((16, l, l))
    # feature = np.zeros((8,l,l))
    if l >= 500:
        # contact_adj = np.zeros((l, l))
        #contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
        #contact = contact_adj
        seq_adj = np.zeros((l, 4))
        seq_adj[:data_len] = data_seq[:data_len]
        data_seq = seq_adj
    for n, cord in enumerate(perm):
        i, j = cord
        data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1), data_seq[:data_len, j].reshape(1, -1))
    data_fcn_1 = np.zeros((1,l,l))
    data_fcn_1[0,:data_len,:data_len] = creatmat(data_seq[:data_len,])

    data_fcn_2 = np.concatenate((data_fcn,data_fcn_1),axis=0)
    # Expand data_seq to match the length of data_fcn_2 along axis 0
    if data_seq.shape[0] < l:
        padded_seq = np.zeros((l, data_seq.shape[1]))
        padded_seq[:data_seq.shape[0], :] = data_seq
        data_seq = padded_seq
    return data_fcn_2, data_len, data_seq
def seq_to_onehot(seq):
    mapping = {'A':0, 'U':1, 'C':2, 'G':3}
    arr = np.zeros((len(seq), 4))
    for i, nt in enumerate(seq):
        arr[i, mapping[nt]] = 1
    return arr
class UFoldPredictor(SecondaryStructurePredictor):
    def __init__(self, model_path="models/ufold_train.pt"):
        super().__init__()
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.contact_net = FCNNet(img_ch=17)
        self.contact_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.contact_net.to(self.device)
        self.contact_net.eval()

    def predict_ss(self, sequence: str):
        # Convert sequence to input embedding
        sequence_onehot = seq_to_onehot(sequence)
        seq_embedding, l, seq_data = get_input_data(sequence_onehot)
        # Pad to length l if needed (optional, here just use actual length)
        seq_embedding = torch.Tensor(seq_embedding).unsqueeze(0).to(self.device)
        seq_data = torch.Tensor(seq_data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_contacts = self.contact_net(seq_embedding)
            pred_contacts = postprocess(pred_contacts, seq_data, 0.01, 0.1, 100, 1.6, True, 1.5)
            contact_map = (pred_contacts > 0.5).float()

        # Convert contact map to base pairs without using get_ct_dict_fast
        contact_map_np = contact_map.cpu().numpy().squeeze()
        base_pairs = []
        n = contact_map_np.shape[-1]
        for i in range(n):
            for j in range(n):
                if contact_map_np[i, j] == 1:
                    base_pairs.append((i + 1, j + 1))
        return base_pairs



if __name__ == "__main__":
    # Example RNA sequence (A, U, C, G)
    # Here we use a simple sequence: "AUGCUAGCUAGC"
    # You need to convert it to one-hot encoding: A=0, U=1, C=2, G=3


    sequence_str = "AGCCCCUAGCGCAGACGGCGGAGAGCAGAGAGGGAGCGCGCCUUGGCUCGCUGGCCUUGGCGGCGGCUCCUCAGGAGAGCUGGGGCGCCCACGAGAGGAUCCCUCACCCGGGUCUCUCCUCAGGGAUGACAUCAUCCGUCCACCUCCUUGUCUUCAAGGACCACCUCCUCUCCAUGCUGAGCUGCUGCCAAGGGGCCUGCUGCCCAUCUACACCUCACGAGGGCACUAGGAGCACGGUUUCCUGGAUCCCACCAACAUACAAAGCAGCCACUCACUGACCCCCAGGACCAGGAUGGCAAAGGAUGAAGAGGACCGGAACUGACCAGCCAGCUGUCCCUCUUACCUAAAGACUUAAACCAAUGCCCUAGUGAGGGGGCAUUGGGCAUUAAGCCCUGACCUUUGCUAUGCUCAUACUUUGACUCUAUGAGUACUUUCCUAUAAGUCUUUGCUUGUGUUCACCUGCUAGCAAACUGGAGUGUUUCCCUCCCCAAGGGGGUGUCAGUCUUUGUCGACUGACUCUGUCAUCACCCUUAUGAUGUCCUGAAUGGAAGGAUCCCUUUGGGAAAUUCUCAGGAGGGGGACCUGGGCCAAGGGCUUGGCCAGCAUCCUGCUGGCAACUCCAAGGCCCUGGGUGGGCUUCUGGAAUGAGCAUGCUACUGAAUCACCAAAGGCACGCCCGACCUCUCUGAAGAUCUUCCUAUCCUUUUCUGGGGGAAUGGGGUCGAUGAGAGCAACCUCCUAGGGUUGUUGUGAGAAUUAAAUGAGAUAAAAGAGGCCUCAGGCAGGAUCUGGCAUAGAGGAGGUGAUCAGCAAAUGUUUGUUGAAAAGGUUUGACAGGUCAGUCCCUUCCCACCCCUCUUGCUUGUCUUACUUGUCUUAUUUAUUCUCCAACAGCACUCCAGGCAGCCCUUGUCCACGGGCUCUCCUUGCAUCAGCCAAGCUUCUUGAAAGGCCUGUCUACACUUGCUGUCUUCCUUCCUCACCUCCAAUUUCCUCUUCAACCCACUGCUUCCUGACUCGCUCUACUCCGUGGAAGCACGCUCACAAAGGCACGUGGGCCGUGGCCCGGCUGGGUCGGCUGAAGAACUGCGGAUGGAAGCUGCGGAAGAGGCCCUGAUGGGGCCCACCAUCCCGGACCCAAGUCUUCUUCCUGGCGGGCCUCUCGUCUCCUUCCUGGUUUGGGCGGAAGCCAUCACCUGGAUGCCUACGUGGGAAGGGACCUCGAAUGUGGGACCCCAGCCCCUCUCCAGCUCGAAAUCCCUCCACAGCCACGGGGACACCCUGCACCUAUUCCCACGGGACAGGCUGGACCCAGAGACUCUGGACCCGGGGCCUCCCCUUGAGUAGAGACCCGCCCUCUGACUGAUGGACGCCGCUGACCUGGGGUCAGACCCGUGGGCUGGACCCCUGCCCACCCCGCAGGAACCCUGAGGCCUAGGGGAGCUGUUGAGCCUUCAGUGUCUGCAUGUGGGAAGUGGGCUCCUUCACCUACCUCACAGGGCUGUUGUGAGGGGCGCUGUGAUGCGGUUCCAAAGCACAGGGCUUGGCGCACCCCACUGUGCUCUCAAUAAAUGUGUUUCCUGUCUUAACAAAAA"


    predictor = UFoldPredictor(model_path="models/ufold_train.pt")
    base_pairs = predictor.predict_ss(sequence_str)

    print("Predicted base pairs:")
    print(base_pairs)