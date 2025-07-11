import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

# from FCN import FCNNet
from Network import U_Net as FCNNet
#from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset,RNASSDataGeneratorPartition
#from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess # 允许某些非规范配对（Non-Canonical Base Pairing）
else:
    from ufold.postprocess import postprocess_new as postprocess  
def pairs2map( pairs, seq_len):
    contact = np.zeros([seq_len, seq_len])
    for pair in pairs:
        contact[pair[0], pair[1]] = 1
    return contact
def read_bpseq(file_path):
    sequence = []
    pairings = []
    with open(file_path, 'r') as f:
        for line in f:
            if(line.strip()==''):
                continue
            pos, base, pair = line.strip().split()
            sequence.append(base)
            if int(pair) != 0:
                pairings.append((int(pos)-1,int(pair)-1))  # 是完全对称的
    return ''.join(sequence), pairings

def structure_combine(predict_sub_bpseq_root_path,truth_bpseq_root_path ,predict_bpseq_root_path):  #合并多个片段为完整的  path_i为包含多个文件的seq1#100.bpseq ，   合并的后的文件存放的文件夹
    # path2_file2_i = ' '  # 完整的真实结构文件夹

    files_exp = [file for file in os.listdir(predict_sub_bpseq_root_path) if file.endswith('.bpseq')]
    files = [x.split('#')[0] for x in files_exp]
    os.makedirs(predict_bpseq_root_path, exist_ok=True)

    # 文件名构成
    df_file = pd.DataFrame()
    df_file['file'] = files
    df_file['file_exp'] = files_exp

    files_uniq = df_file.drop_duplicates(subset=['file'])
    files_uniq = files_uniq['file']

    for file_uniq in files_uniq:
        df_tem = df_file[df_file['file'] == file_uniq]
        # file_uniq = file_uniq.split('.')[0]
        df_bpseq = pd.read_table(os.path.join(truth_bpseq_root_path ,file_uniq + '.bpseq'), header=None, sep=' ',
                                 names=['num', 'base', 'to'])
        df_bpseq['to_after_par'] = 0

        for file in [os.path.join(predict_sub_bpseq_root_path ,file_exp) for file_exp in df_tem['file_exp']]:
            df_bpseq_sub = pd.read_table(file, header=None, sep=' ', names=['num', 'base', 'to'])
            offset = int(os.path.basename(file).split('#')[1].split('.')[0])-1  # Fix
            len_sub = df_bpseq_sub.shape[0]

            real_pos = [x for x in range(offset, offset + len_sub)]
            real_to = df_bpseq_sub['to']
            real_to[real_to != 0] = offset + real_to[real_to != 0]

            to_after_par = df_bpseq['to_after_par']
            to_after_par.iloc[real_pos] = real_to
            df_bpseq['to_after_par'] = to_after_par

        df_bpseq = df_bpseq.drop('to', axis=1)
        df_bpseq.to_csv(os.path.join(predict_bpseq_root_path , file_uniq + '.bpseq'), sep=' ', index=False, header=False)

def contact2bpseq(pred_matrix, seq, seq_len):
    """将接触矩阵转换为bpseq格式"""
    pairs = np.where(pred_matrix > 0.5)  # 阈值可根据模型调整
    # pair_dict = {i+1:j+1 for i,j in zip(*pairs) if i < j}  # 1-based坐标
    pair_dict = {i+1:j+1 for i,j in zip(*pairs) }  # 1-based坐标
    
    # 构建bpseq数据框架
    df = pd.DataFrame({
        'num': range(1, seq_len+1),
        'base': list(seq),
        'to': [pair_dict.get(i,0) for i in range(1, seq_len+1)]
    })
    return df


def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation



def eval_result(complete_predict_root_path,truth_ss_root_path,result_save_path,pretrained_weight_name):
    
    predict_complete_files= [f for f in os.listdir(complete_predict_root_path) if f.endswith(".bpseq")]
    result_list=[]
    
    
    tp_list=[]
    tn_list=[]
    fp_list=[]
    fn_list=[]
    accuracy_list=[]
    prec_list=[]
    recall_list=[]
    F1_list=[]
    MCC_list=[]
    sens_list=[]
    spec_list=[]
    total_name_list=[]
    total_length_list=[]
    for bpseq_filename in predict_complete_files:
        predict_path = os.path.join(complete_predict_root_path, bpseq_filename)
        truth_path = os.path.join(truth_ss_root_path, bpseq_filename)
        base_name = os.path.splitext(bpseq_filename)[0]
        
        pred_seq,pred_pairs=read_bpseq(predict_path)
        pred_contact=pairs2map(pred_pairs, len(pred_seq))
        truth_seq,truth_pairs=read_bpseq(truth_path)
        truth_contact=pairs2map(truth_pairs, len(truth_seq))
        assert pred_seq==truth_seq
        # 计算性能
        (tp,tn,fp,fn,accuracy, prec, recall, sens, spec, F1, MCC )=rna_evaluation_modified(
            torch.tensor(pred_contact),torch.tensor(truth_contact))
        
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        accuracy_list.append(accuracy)
        prec_list.append(prec)
        recall_list.append(recall)
        F1_list.append(F1)
        MCC_list.append(MCC)
        sens_list.append(sens)
        spec_list.append(spec)
        total_name_list.append(base_name)
        total_length_list.append(len(pred_seq))
        
        
    result_df = pd.DataFrame({'id': total_name_list,
                                'len': total_length_list,
                                'TP' : list(np.array(tp_list)),
                                'TN' : list(np.array(tn_list)),
                                'FP' : list(np.array(fp_list)),
                                'FN' : list(np.array(fn_list)),
                                'acc': list(np.array(accuracy_list)),
                                'pre': list(np.array(prec_list)),
                                'rec': list(np.array(recall_list)),
                                'mcc': list(np.array(MCC_list)),
                                'F1': list(np.array(F1_list)),
                                'sensitivity': list(np.array(sens_list)),
                                'specificity': list(np.array(spec_list))})
    result_df.to_csv(os.path.join(result_save_path, f'{pretrained_weight_name}_bpRNA_new_result_ufold_evaluation.csv'), index=False, header=True)

def model_eval_all_test(contact_net,test_generator,predict_result_path=''):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device( "cpu")
    contact_net.train()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    res_list=[]
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)  #0.05左右
    #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in test_generator:
    for seq_embeddings,seq_raw, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:  # nc_map：根据nc_perm判断是否存在配对
    # data_fcn_2,seq_raw, matrix_rep, data_len, data_seq[:l], data_name, data_nc,l
        # nc_map_nc = nc_map.float() * contacts
        if seq_lens.item() > 1500:
            continue
        if batch_n%1000==0:
            print('Batch number: ', batch_n)
        #if batch_n > 3:
        batch_n += 1
        #    break
        #if batch_n-1 in rep_ind:
        #    continue
        # contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)  # 1 17 80 80
        ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)  #1 80 4
        # matrix_reps_batch = torch.unsqueeze(
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())  # 34
        #     torch.Tensor(matrix_reps.float()).to(device), -1)

        # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
        #     seq_len, seq_len]).to(device)

        # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
        tik = time.time()
        
        with torch.no_grad():
            #pred_contacts = contact_net(seq_embedding_batch,seq_embedding_batch_1)
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,   # 1 80 80  # 通过梯度下降和拉格朗日乘子法，将原始预测（可能来自深度学习模型）调整为符合生物物理规则的接触矩阵。其核心是在效用最大化和结构约束之间寻求平衡，适用于RNA结构预测或序列设计任务。
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6    1 80 4
            #seq_ori, 0.01, 0.1, 100, 1.6, True) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train  # 1 80 80 使用nc进行二次约束
        map_no_train = (u_no_train > 0.5).float()  # 1 80 80  用这个
        # map_no_train_nc = (nc_no_train > 0.5).float()
        
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        assert len(map_no_train)==1 #batchsize必须为1
        
        
                    # 把每一条数据保存到指定的路径 predict_save_path
        for i in range(map_no_train.shape[0]):
            pred_matrix = map_no_train[i].squeeze(0).cpu().numpy()
            assert pred_matrix.shape[0] == pred_matrix.shape[1]
            seq_length = seq_lens[i].item()
            seq_name = seq_name[i]
            seq = seq_raw[i]
            
            # Convert the contact matrix to bpseq format
            df_bpseq = contact2bpseq(pred_matrix, seq, seq_length)
            
            # Save the bpseq file
            save_path = os.path.join(predict_result_path, f"{seq_name}.bpseq")
            df_bpseq.to_csv(save_path, sep=' ', index=False, header=False)

        


    # result_df = pd.DataFrame(res_list)
    # result_df.to_csv(os.path.join('./results', f'ufold_train_bpRNA_new_result.csv'), index=False, header=True)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(0)
    
    #pdb.set_trace()
    
    config_file = args.config
    test_file = args.test_files  # 数据集名称
    
    config = process_config(config_file)

    MODEL_SAVED = 'models/ufold_train.pt'

    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1  # 1
    OUT_STEP = config.OUT_STEP  # 100
    LOAD_MODEL = config.LOAD_MODEL  # True
    data_type = config.data_type
    model_type = config.model_type
    model_path = '/data2/darren/experiment/ufold/models_ckpt/'.format(model_type, data_type,d)
    epoches_first = config.epoches_first  # 100 epoch ？
    
    
    # if gpu is to be used
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    seed_torch()
    
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
    
    # train_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'train', True)
    # val_data = RNASSDataGenerator('/home/yingxic4/programs/e2efold/data/{}/'.format(data_type), 'val')
    ##test_data = RNASSDataGenerator('./data/{}/'.format(data_type), 'test_no_redundant.pickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant.pickle')
    #test_data = RNASSDataGenerator('/data2/darren/experiment/ufold/data/rnastralign_all/', 'test_no_redundant_600.pickle')
    print('Loading test file: ',test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    elif test_file == 'bpRNAnew_partition':
        test_data = RNASSDataGeneratorPartition('/local4/local_dataset/RNA_BenchMark/SS/data/xz/fyf_for_Ufold/bpRNAnew_partition.pkl')
    else:
        test_data = RNASSDataGenerator('data/',test_file+'.cPickle')
    
    seq_len = test_data.data_x.shape[-2]  # 600
    print('Max seq length ', seq_len)
    #pdb.set_trace()
    
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}
    # # train_set = Dataset(train_data)
    # train_set = Dataset_FCN(train_data)
    # train_generator = data.DataLoader(train_set, **params)
    
    # # val_set = Dataset(val_data)
    # val_set = Dataset_FCN(val_data)
    # val_generator = data.DataLoader(val_set, **params)
    
    # test_set = Dataset(test_data)
    # test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_data, **params)
    
    '''
    test_merge = Dataset_FCN_merge(test_data,test_data2)
    test_merge_generator = data.DataLoader(test_merge, **params)
    pdb.set_trace()
    '''
    
    
    
    contact_net = FCNNet(img_ch=17)  # 实际是一个U-Net
    
    #pdb.set_trace()
    print('==========Start Loading==========')
    # contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:0'))
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cpu'))
    print('==========Finish Loading==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    truth_ss_root_path='/local4/local_dataset/RNA_BenchMark/SS/data/xz/bpRNAnew_partition/truth_bpseq'
    sub_predict_result_path='/local4/fyf/TEST/UFold/results/predict_result/sub_predict'
    complete_predict_root_path='/local4/fyf/TEST/UFold/results/predict_result/complete_predict'
    model_eval_all_test(contact_net,test_generator,sub_predict_result_path)
    
    structure_combine(sub_predict_result_path,truth_ss_root_path ,complete_predict_root_path)
    eval_result(complete_predict_root_path,truth_ss_root_path,result_save_path,pretrained_weight_name)

    
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    '''
    字段名	        类型	        典型内容示例	                说明
    seq	            str	            "AUGCCGGA"	                RNA 的核苷酸序列（如 A, U, C, G）
    ss_label	    str	            "(((...)))"             	二级结构标签（用符号表示碱基配对情况）
    length	        int	            20	                        RNA 序列的长度
    name	        str	            "RNA_001"	                RNA 序列的名称或唯一标识符
    pairs	        list	        [(0, 5), (1, 4), (2, 3)]	碱基配对的具体位置索引（配对列表）
    
    
    '''
    main()






