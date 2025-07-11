# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
from prediction_utils import *
from tqdm import tqdm
# from xz_evaluate import structure_combine
from common.loss_utils import rna_evaluation_modified
from eval_utils import parse_config, get_data_test, get_model_test, vote4struct, clean_dict, log_eval_metrics, \
    save_metrics
import pickle
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


def evaluation(args, eval_model, dataloader,predict_save_path): 
    eval_model.eval()
    device = args.device
    predict_result=[]

    with torch.no_grad():
        test_no_train = list()
        total_name_list = list()
        total_length_list = list()
        for _, (seq_raw_list,data_fcn_2, tokens, data_length, data_name, set_max_len, data_seq_encoding) in enumerate(dataloader):
            total_name_list += [item for item in data_name]
            total_length_list += [item.item() for item in data_length]
            batch_size = tokens.shape[0]

            data_fcn_2 = data_fcn_2.to(device)
            matrix_rep = torch.zeros(batch_size, set_max_len, set_max_len).unsqueeze(1)
            data_length = data_length.to(device)
            tokens = tokens.to(device)
            data_seq_encoding = data_seq_encoding.to(device)
            contact_masks = contact_map_masks(data_length, matrix_rep).to(device)

            # for multi conformations sampling
            pred_x0_copy_dict = dict()
            best_pred_x0_i_list = list()
            candidate_seeds = np.arange(0, 2023)
            select_seeds = np.random.choice(candidate_seeds, args.num_samples).tolist()
            for seed_ind in select_seeds:
                torch.manual_seed(seed_ind)

                pred_x0, _ = eval_model.sample(batch_size, data_fcn_2, tokens, set_max_len, contact_masks, data_seq_encoding)
                pred_x0_copy_dict[seed_ind] = pred_x0
            # 32 1 160 160
            for i in tqdm(range(pred_x0.shape[0]), desc=f'vote for the most common structure', total=pred_x0.shape[0]):
                pred_x0_i_list = [pred_x0_copy_dict[num_copy][i].squeeze().cpu().numpy() for num_copy in select_seeds]  # 10 160 160  复制10份
                best_pred_x0_i = torch.Tensor(vote4struct(pred_x0_i_list))  # 160 160
                # best_pred_x0_i = posprocess(best_pred_x0_i, data_length[i].item())  # 160 160
                best_pred_x0_i_list.append(best_pred_x0_i)

            pred_x0 = torch.stack(best_pred_x0_i_list, dim=0)
            pred_x0 = pred_x0.cpu().float().unsqueeze(1)
 
            # 把每一条数据保存到指定的路径 predict_save_path
            for i in range(pred_x0.shape[0]):
                pred_matrix = pred_x0[i].squeeze().cpu().numpy()
                seq_length = data_length[i].item()
                seq_name = data_name[i]
                seq = seq_raw_list[i]
                
                # Convert the contact matrix to bpseq format
                df_bpseq = contact2bpseq(pred_matrix, seq, seq_length)
                
                # Save the bpseq file
                save_path = os.path.join(predict_save_path, f"{seq_name}.bpseq")
                df_bpseq.to_csv(save_path, sep=' ', index=False, header=False)




def prediction(config, model, data_fcn_2, tokens, seq_encoding_pad, seq_length, set_max_len):
    device = config.device
    model.to(device)
    model.eval()
    with torch.no_grad():
        data_fcn_2 = data_fcn_2.to(device)
        tokens = tokens.to(device)
        seq_encoding_pad = seq_encoding_pad.to(device)
        seq_length = seq_length.to(device)
        batch_size = data_fcn_2.shape[0]
        matrix_rep = torch.zeros((batch_size, set_max_len, set_max_len)).unsqueeze(1)
        contact_masks = contact_map_masks(seq_length, matrix_rep).to(device)

        # for multi conformations sampling
        pred_x0_copy_dict = dict()
        best_pred_x0_i_list = list()
        candidate_seeds = np.arange(1970, 2023)
        candidate_seeds = np.arange(0, 2023)
        select_seeds = np.random.choice(candidate_seeds, config.num_samples).tolist()
        for seed_ind in select_seeds:
            torch.manual_seed(seed_ind)

            pred_x0, _ = model.sample(batch_size, data_fcn_2, tokens,
                                      set_max_len, contact_masks, seq_encoding_pad)
            pred_x0_copy_dict[seed_ind] = pred_x0

        for i in tqdm(range(pred_x0.shape[0]), desc=f'vote for the most common structure', total=pred_x0.shape[0]):
            pred_x0_i_list = [pred_x0_copy_dict[num_copy][i].squeeze().cpu().numpy() for num_copy in select_seeds]
            best_pred_x0_i = torch.Tensor(vote4struct(pred_x0_i_list))
            best_pred_x0_i_list.append(best_pred_x0_i)

    return best_pred_x0_i_list
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


def structure_performance_single_file(path_pred, path_real, eps=1e-10):
    fasta_name = path_pred.split('/')[-1].split('.')[0]
    
    # 读取文件并处理空值
    df_pred = pd.read_table(path_pred, sep=' ', names=['from', 'base', 'to'])
    df_real = pd.read_table(path_real, sep=' ', header=None, names=['from', 'base', 'to'])
    df_pred.fillna(0, inplace=True)

    # 初始化混淆矩阵
    TP, TN, FP, FN = 0, 0, 0, 0
    
    # 配对验证逻辑（考虑对称配对）
    for i in range(len(df_pred)):
        pred_to = int(df_pred.iloc[i]['to'])
        real_to = int(df_real.iloc[i]['to'])
        
        # 处理对称配对有效性（i-j 与 j-i 等价）
        reciprocal_check = (pred_to == df_real.iloc[real_to-1]['to']) if real_to !=0 else False
        
        if pred_to != 0:
            if pred_to == real_to or reciprocal_check:
                TP += 1
            else:
                FP += 1
        else:
            if real_to == 0:
                TN += 1
            else:
                FN += 1

    # 带eps保护的指标计算
    denominator_acc = TP + TN + FP + FN + eps
    denominator_pre = TP + FP + eps
    denominator_rec = TP + FN + eps
    denominator_f1 = (denominator_pre + denominator_rec) + eps
    
    acc = (TP + TN) / denominator_acc
    pre = TP / denominator_pre
    rec = TP / denominator_rec
    f1 = (2 * pre * rec) / (pre + rec + eps) if (pre + rec) > 0 else 0
    
    # 改进的MCC计算（符合原始公式）
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = math.sqrt(
        (TP + FP + eps) * 
        (TP + FN + eps) * 
        (TN + FP + eps) * 
        (TN + FN + eps)
    ) + eps
    mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
    # id,tp,tn,fp,fn,accuracy, prec, recall, sens, spec, F1, MCC  
    return {
        'id': fasta_name,
        'tp': TP,
        'tn': TN,
        'fp': FP,
        'fn': FN,
        'accuracy': round(acc, 4),
        'prec': round(pre, 4),
        'recall': round(rec, 4),
        'sens': round(rec, 4),
        'spec': round(TN/(TN+FP), 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4),
        'length': len(df_pred),

    }
if __name__ == '__main__':
    # 新路径配置
    # test_partition的测试
    # seq_root_path = "/local4/local_dataset/RNA_BenchMark/SS/data/xz/ts/fasta"          # 输入FASTA目录
    # predict_root_path = "/local4/fyf/TEST/RNADiffFold/predict_results/xz_predict_parition"      # 预测bpseq输出目录 
    # truth_ss_root_path = "/local4/local_dataset/RNA_BenchMark/SS/data/xz/ts/bpseq"    # 真实结构目录
    # data_path ='/local4/local_dataset/RNA_BenchMark/SS/data/UFold/fyf/bpnew.pkl'
    # partition_seq_root_path = "/local4/local_dataset/RNA_BenchMark/SS/data/xz/ts/fasta_partition"  # 分割序列目录
    # bpRNA_new partition 的测试
    # seq_root_path = "/local4/local_dataset/RNA_BenchMark/SS/data/xz/bpRNA_new_partition/fasta_par"          # 输入FASTA目录
    partition_seq_root_path = "/local4/local_dataset/RNA_BenchMark/SS/data/xz/bpRNA_new_partition/fasta_par"  # 分割序列目录
    predict_root_path = "/local4/fyf/TEST/RNADiffFold/predict_results/bpRNA_new_predict_parition"      # 预测bpseq输出目录 
    truth_ss_root_path = "/local4/local_dataset/RNA_BenchMark/SS/data/xz/bpRNA_new_partition/truth_bpseq"    # 真实结构目录
    
    # data=pickle.load(data_path, 'rb')
    
    
    sub_predict_root_path = os.path.join(predict_root_path,"predict_sub")  # 子预测bpseq输出目录
    complete_predict_root_path = os.path.join(predict_root_path,"predict_complete")  # 子预测bpseq输出目录

    
    result_save_path ='/local4/fyf/TEST/RNADiffFold/predict_results'
    evaluation_way='RNADiffFold'  # xz or RNADiffFold
    already_complete_predict=True  # 是否已经完成了完整bpseq的预测
    # 创建预测目录
    os.makedirs(predict_root_path, exist_ok=True)
    os.makedirs(sub_predict_root_path, exist_ok=True)
    os.makedirs(complete_predict_root_path, exist_ok=True)
    
    config = parse_config(join(os.getcwd(), 'config_evaluation_bpRNA_new.json'))
    set_seed(config.seed)

    model, alphabet = get_model_prediction(config.model)
    
    
    test_loader = get_data_test(config.data, alphabet)

    # model load checkpoint
    print(f"Load model checkpoint from: {config.model_ckpt_path}")
    checkpoint = torch.load(config.model_ckpt_path, map_location='cpu')
    # model_state_dict = torch.load(config.model_state_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # model.load_state_dict(model_state_dict, strict=True)
    model.to(config.device)
    
    if not already_complete_predict:
        evaluation(config, model, test_loader,sub_predict_root_path)

        structure_combine(sub_predict_root_path,truth_ss_root_path ,complete_predict_root_path)  # 合并多个片段为完整的bpseq文件
    # # structure_combine(sub_predict_root_path,truth_ss_root_path ,complete_predict_root_path)  # 合并多个片段为完整的bpseq文件
    
    # res_dict_list=[]
    # 计算性能
    # id_list=[]
    
    if evaluation_way == 'xz':
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
        predict_complete_files= [f for f in os.listdir(complete_predict_root_path) if f.endswith(".bpseq")]
        for predict_complete_file in predict_complete_files:
            base_name = os.path.splitext(predict_complete_file)[0]
            predict_path = os.path.join(complete_predict_root_path, predict_complete_file)
            truth_path = os.path.join(truth_ss_root_path, f"{base_name}.bpseq")
            # # id,tp,tn,fp,fn,accuracy, prec, recall, sens, spec, F1, MCC  
            # id,tp,tn,fp,fn,accuracy, prec, recall, sens, spec, F1, MCC,length = structure_performance_single_file(predict_path, truth_path)
            performance_res= structure_performance_single_file(predict_path, truth_path)
            # res_dict_list.append(res_dict)
            # id_list.append(id)
            tp_list.append(performance_res['tp'])
            tn_list.append(performance_res['tn'])
            fp_list.append(performance_res['fp'])
            fn_list.append(performance_res['fn'])
            accuracy_list.append(performance_res['accuracy'])
            prec_list.append(performance_res['prec'])
            recall_list.append(performance_res['recall'])
            F1_list.append(performance_res['F1'])
            MCC_list.append(performance_res['MCC'])
            sens_list.append(performance_res['sens'])
            spec_list.append(performance_res['spec'])
            total_name_list.append(performance_res['id'])
            total_length_list.append(performance_res['length'])
            
            



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
        result_df.to_csv(os.path.join(result_save_path, f'{config.data.dataset}_result_xz_evaluation.csv'), index=False, header=True)
    else:
        
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
        result_df.to_csv(os.path.join(result_save_path, f'{config.data.dataset}_result_rnadifffold_evaluation.csv'), index=False, header=True)
    
    
            
            



