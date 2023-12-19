import os
import datasets
import pickle
from srrtransformer.utils import SimulateErrors, batch_normalized_hamming_distance,seq2splits
import srrtransformer.DNAmodel as DNAmodel
from srrtransformer.data_utils import Data2Dataset
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from itertools import product
import pandas as pd
from tqdm import tqdm
import numpy as np
import edist.sed
k_mer_size = config["k_mer_size"]
seq_length = config["seq_length"]
data_type = config["data_type"]


def FilterData(df,task=None,seq_length=None,do_print=False):
        """
    *Filter data strings in a data frame according to their length with respect to seq_length    
    SL: Short Length
    LL: Long Length
    CL: Correct Length *
    
        parameters:
        ----------
            :param df:
                type: pd.dataframe
                A data frame with required fields: dna_copy_len
            :param task:
                type: str
                Defult: none
                valid inputs: LL (Long Length), SL (Short Length), CL (Correct Length) 
            :param seq_length:
                type: int
                Defult: none
            :param do_print:
                type: boolean
                Defult: False
                Prints details about the new filtered dataframe
                

        returns:
        --------
        Filtered DF according to the length

    """
    orig_length = len(df)
    if seq_length:
        seq_length = seq_length
    else:
        seq_length = USER_CONFIG["seq_length"]
    if task == "LL":
        df = df.loc[df["dna_copy_len"]>seq_length].copy()
    elif task == "SL":
        df = df.loc[df["dna_copy_len"]<seq_length].copy()
    elif task == "CL":
        df = df.loc[df["dna_copy_len"]==seq_length].copy()
        
        """
        elif task == "TC":
            df = df.loc[df["is_error"]==0].copy()
        """  
    else:
        task = "-"
        df = df.copy()
        
    df['task'] = task
    if do_print:
        print(task,": "," size: ",len(df)," length mean: ",round(df["dna_copy_len"].mean(),4),
              " HammingScore: ",round(df["HammingScore"].mean(),4)," ratio from orig: ",round((len(df)/orig_length),4))
    return df

def SplitCL2Realible(df):
        """
    *Splits Correct Length sequences to realible (with only valid codewords) and not_realiable (with non valid codewords)
    We are aware of the spelling mistake, it is more fun this way*
    
        parameters:
        ----------
            :param df:
                type: pd.dataframe
                A data frame with required fields: orig_DNA, dna_copy

                

        returns:
        --------
       columns of realiable and not_realiable sequences

    """
    data = df.copy()
    k_mer_size = config["k_mer_size"]
    seq_length = config["seq_length"]
    codewords = [''.join(i) for i in product(["A","C","T","G"], repeat = k_mer_size)]
    data['splited_orig'] = [seq2splits(seq,k_mer_size) for seq in data["orig_DNA"]]
    strings = " ".join(data['splited_orig'].to_list()).split(" ")
    realible_codewords = list(pd.Series(strings).unique())
    print(len(realible_codewords))
    error_codewords = [i for i in codewords if i not in realible_codewords]
    print(len(error_codewords))
    error_codewords_str = "|".join(error_codewords)
    data['splited_copy'] = [seq2splits(seq,k_mer_size) for seq in data["dna_copy"]]

    realible_length_reads = data.loc[data.dna_copy_len==seq_length].copy()

    realible_data = realible_length_reads.loc[~(realible_length_reads.splited_copy.str.contains(error_codewords_str))].copy()
    not_realible_data = realible_length_reads.loc[realible_length_reads.splited_copy.str.contains(error_codewords_str)].copy()

    return realible_data,not_realible_data


def SaveRealibleCodewords(df):
            """
    *Saves the realible codewords used to encode the file described in the df*
    
        parameters:
        ----------
            :param df:
                type: pd.dataframe
                A data frame with required fields: orig_DNA

                

        returns:
        --------
       List of the realiable codewords in data frame, and the "cost" to save them

    """
    data = df.copy()
    total_bases_in_data = int(len(data)*seq_length*k_mer_size)
    data['splited_orig'] = [seq2splits(seq,k_mer_size) for seq in data["orig_DNA"]]
    strings = " ".join(data['splited_orig'].to_list()).split(" ")
    realible_codewords = list(pd.Series(strings).unique())
    
    print("num of realible codewords: ",len(realible_codewords))
    cost_to_write = (len(realible_codewords)*k_mer_size)/total_bases_in_data
    print("dictionary writing cost: ",round(cost_to_write*100,6),"%")
    return cost_to_write,realible_codewords

def Realible2UnrealibleCodewords(realible_codewords):
                """
    *Given a list of realiable codewords, returns the unrealible*
    
        parameters:
        ----------
            :param realible_codewords:
                type: list 
                A list of strings with the realible codewords (strings over ATCG)

                

        returns:
        --------
       List of the not_realiable codewords 

    """
    all_possible_codewords = [''.join(i) for i in product(["A","C","T","G"], repeat = k_mer_size)]
    error_codewords = [i for i in all_possible_codewords if i not in realible_codewords]
    return error_codewords


def GetRealibleData(df,realible_codewords):
                    """
    *Given a df and a list of realible codewords used to encode the data in the df, return the realiable and not realiable data,
    and the list of error codewords*
    
        parameters:
        ----------
            :param df:
                type: pd.dataframe 
                A datafrom with of dna encoded data
            :param realible_codewords:
                type: list 
                A list of strings with the realible codewords (strings over ATCG)


                

        returns:
        --------
       List of the realiabe, not_realiable sequences and the list of unrealible codewords

    """
    data = df.copy()
    error_codewords = Realible2UnrealibleCodewords(realible_codewords)
    error_codewords_str = "|".join(error_codewords)
    
    realible_length_reads = data.loc[data["dna_copy_len"]==seq_length].copy()
    realible_length_reads['splited_copy'] = [seq2splits(seq,k_mer_size) for seq in realible_length_reads["dna_copy"]]
    realible_data = realible_length_reads.loc[~(realible_length_reads["splited_copy"].str.contains(error_codewords_str))].copy()
    not_realible_data = realible_length_reads.loc[realible_length_reads["splited_copy"].str.contains(error_codewords_str)].copy()

    return realible_data,not_realible_data,error_codewords

def batch2splits(seq, k):
                        """
    *Splits a list/string * 
    
        parameters:
        ----------
            :param seq:
                type: list/str.
                the list/str wished to split
            :param k:
                type: int
                the size of each split


        returns:
        --------
    A splited list\string
        
        
        """
    chunks = len(seq)
    chunk_size = k
    split_seq = [seq[i:i+chunk_size] for i in range(0,chunks,chunk_size)]
    return split_seq

def detailed_edit_distance(string_b,string_a): ### Function that takes 2 strings and returns the detailed num of operations to get from x to y
                    """
    * takes 2 strings and returns the detailed num of edit operations to get from x to y*
    
        parameters:
        ----------
            :param string_b:
                type: str 
            :param string_a:
                type: str


                

        returns:
        --------
       number of substituition, number of deletions, number of insertions
       """

    total_distance = edist.sed.sed_string(string_a, string_b)
    alignment = edist.sed.standard_sed_backtrace(string_a, string_b)
    num_of_insertions = 0
    num_of_deletions = 0
    for align in alignment:
        align = str(align)
        #print(align)
        if align[len(align)-1] == "-":
          #print("del")
          num_of_deletions=num_of_deletions+1
        if align[0] == "-":
          #print("ins")
          num_of_insertions=num_of_insertions+1
    num_of_substitutions = total_distance - num_of_insertions - num_of_deletions
    return (num_of_substitutions,num_of_deletions,num_of_insertions)

def edit_distance_summ(column_a=None,column_b=None):
                    """
    * Given two columns of strings, return the total number of edit operations to get from strings in column a to column b*
    
        parameters:
        ----------
            :param column_a:
                type: df column of strings 
                Default: None
            :param column_b:
                type: df column of strings 
                Default: None


                

        returns:
        --------
       Total number of substituition, deletions,insertions and the edit error rate
       """
    substitutions = 0
    deletions = 0
    insertions = 0    
    len_df = len(column_a)
    
    for string_a,string_b in zip(list(column_a),list(column_b)):
        edit_dist = detailed_edit_distance(string_a,string_b)
        substitutions  =  substitutions  + edit_dist[0]
        deletions =  deletions + edit_dist[1]
        insertions =  insertions + edit_dist[2]
    
    subs = (substitutions/len_df)/len(string_a)
    dels = (deletions/len_df)/len(string_a)
    ins = (insertions/len_df)/len(string_a)
    error_rate = 1-((1-subs)*(1-dels)*(1-ins))
    
    subs = round(subs,4)
    dels = round(dels,4)
    ins = round(ins,4)
    error_rate = round(error_rate,4)
    
    return subs,dels,ins,error_rate

def list2splits(seq, k):
                            """
    *Splits a list * 
    
        parameters:
        ----------
            :param seq:
                type: List.
                list we wish to split
            :param k:
                type: int.
                size of each split


        returns:
        --------
    A list split to chunk of size k
        
        
        """
    chunks = len(seq)
    chunk_size = k
    split_seq = [seq[i:i+chunk_size] for i in range(0,chunks,chunk_size)]
    return split_seq

def print_stats(df,key,orig_length=1):
                            """
    *Prints stats of a dataframe * 
    
        parameters:
        ----------
            :param df:
                type: pd.dataframe
            :param key:
                type: int.
            :param orig_length:
                type: int.
                Default: 1


        returns:
        --------
      Prints stats of a dataframe 
        
        """
    print(key,": "," size: ",len(df)," length mean: ",round(df["dna_copy_len"].mean(),4),
      " HammingScore: ",round(df["HammingScore"].mean(),4)," ratio from orig: ",round((len(df)/orig_length),4))    

def SplitReadsByType(df,realible_codewords=None):
    """
    *Splits a data frame to 
    SL: Short Length
    LL: Long Length
    CL: Correct Length (Bad Tokens Good Size)
    GTGS: Good Tokens Good Size   *
    
            :param df:
                type: pd.dataframe
            :param realible_codewords:
                type: list
                list of realible codewords according to which we split the df
        returns:
        --------
      A dictionary with keys GTGS,SL,LL,CL and a list of error_codewords
        
    """
    orig_length = len(df)
    predictions_sets = {}

    predictions_sets["GTGS"], predictions_sets["CL"], error_codewords = GetRealibleData(df,realible_codewords)
    print("num of error codewords: ",len(error_codewords))
    predictions_sets["SL"] = FilterData(df,"SL")
    predictions_sets["LL"] = FilterData(df,"LL")
    
    print_stats(df,"orig data",orig_length)
    for key in predictions_sets.keys():
        print_stats(predictions_sets[key],key,orig_length)

    return predictions_sets,error_codewords

def MakeGTGSResults(error_df):
        """
    *Predict GTGS with identity mapping and calculate error parameters   *
    
            :param error_df:
                type: pd.dataframe
                A data frame of the GTGS set
        returns:
        --------
      The df with the predictions and the all the errors metrices
        
    """
    pred_str = list(error_df["dna_copy"]) 
    label_str = list(error_df["orig_DNA"])
    pred_hamming_distance = batch_normalized_hamming_distance(pred_str,label_str,return_list=True)
    subs,dels,ins,error_rate = edit_distance_summ(label_str,pred_str)

    error_df['dna_recoverd'] = pred_str
    error_df['pred_hamming_distance'] = pred_hamming_distance
    error_df['pred_is_error'] = (error_df['dna_recoverd']!=error_df["orig_DNA"]).astype(int)  
    HammingScore = error_df['pred_hamming_distance'].mean()
    ErrorScore = error_df['pred_is_error'].mean()
    AccuracyScore = 1-ErrorScore
    print(f"HammingScore: {HammingScore*100} %")
    metrics = {"HammingScore":HammingScore,
                "ErrorScore":ErrorScore,
                "AccuracyScore":AccuracyScore,
                "Edit_error_rate":error_rate,
                "Substitution_rate:":subs,
                "Deletion_rate:":dels,
                "Insertion_rate:":ins,
              }
    return error_df,metrics


def PredictEndDecode(error_df,model_dir,error_codewords=None,batch_size=128):
                    """
    *Predicts and decodes error strings in a data frame*
    
        parameters:
        ----------
            :param error_df:
                type: pd.dataframe 
                A datafrom with of dna encoded data with errors
            :param model_dir:
                type: str 
                the directory of the model
            :param error_codewords:
                type: list
                Default: None
                A list with the error codewords
            :param batch_size:
                type: int 
                Default: 128
                The batch size for prediction


                

        returns:
        --------
       The df with the predictions and the all the errors metrices

    """
    if torch.cuda.device_count()>0:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        
    error_df = error_df.copy()
    dna_model = DNAmodel.DnaModel(batch_size=batch_size,from_pretrained=True,model_dir=model_dir)
    dna_model.load_model()
    
    bad_words_id = None
    if error_codewords:
        print("using error codewords...")
        bad_words_id = dna_model.tokenizer(error_codewords,add_special_tokens=False).input_ids
    
    error_dataset = Data2Dataset(error_df)
    dataset = dna_model.format_data(error_dataset)
    predictions = []
    idx = np.arange(0,len(dataset),batch_size)
    for i in tqdm(idx):
        batch = dataset[i:i+batch_size]
        generation_output = dna_model.model.generate(batch['input_ids'].to(device),min_length=dna_model.decoder_max_length-1,
                                                     max_length=dna_model.decoder_max_length-1,num_beams=5,bad_words_ids=bad_words_id)
        prediction = dna_model.tokenizer.batch_decode(generation_output,skip_special_tokens=True)
        predictions += prediction  

    pred_str = [seq.replace(" ","") for seq in predictions] 
    label_str = list(error_df["orig_DNA"])
    pred_hamming_distance = batch_normalized_hamming_distance(pred_str,label_str,return_list=True)
    subs,dels,ins,error_rate = edit_distance_summ(label_str,pred_str)
    
    error_df['dna_recoverd'] = pred_str
    error_df['pred_hamming_distance'] = pred_hamming_distance
    error_df['pred_is_error'] = (error_df['dna_recoverd']!=error_df["orig_DNA"]).astype(int)
    
    HammingScore = error_df['pred_hamming_distance'].mean()
    ErrorScore = error_df['pred_is_error'].mean()
    AccuracyScore = 1-ErrorScore
    
    print(f"HammingScore: {HammingScore*100} %   Edit_error_rate:{error_rate*100} %")
    metrics = {"HammingScore":HammingScore,
                "ErrorScore":ErrorScore,
                "AccuracyScore":AccuracyScore,
                "Edit_error_rate":error_rate,
                "Substitution_rate:":subs,
                "Deletion_rate:":dels,
                "Insertion_rate:":ins,
              }
    torch.cuda.empty_cache()
    del dna_model
    return error_df,metrics


def SelectivePredictEndDecode(error_df,model_dir,error_codewords=None,batch_size=128,num_beams=15):
                    """
    *Apply constrained beam search for prediction decoding*
    
        parameters:
        ----------
            :param error_df:
                type: pd.dataframe 
                A datafrom with of dna encoded data with errors
            :param model_dir:
                type: str 
                the directory of the model
            :param error_codewords:
                type: list
                Default: None
                A list with the error codewords
            :param batch_size:
                type: int 
                Default: 128
                The batch size for prediction
            :param num_beams:
                type: int 
                Default: 15
                The beam size
         

        returns:
        --------
       The df with the predictions and the all the errors metrices

    """
    print("Running Selective Decoding...")
    if torch.cuda.device_count()>0:
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    error_df = error_df.copy()
    dna_model = DNAmodel.DnaModel(batch_size=batch_size,from_pretrained=True,model_dir=model_dir)
    dna_model.load_model()

    bad_words_id = None
    if error_codewords:
        print("using error codewords...")
        bad_words_id = dna_model.tokenizer(error_codewords,add_special_tokens=False).input_ids

    error_dataset = Data2Dataset(error_df)
    dataset = dna_model.format_data(error_dataset)
    fake_labels = list(error_df["dna_copy"])
    predictions = []
    idx = np.arange(0,len(dataset),batch_size)
    for i in tqdm(idx):
        batch = dataset[i:i+batch_size]
        generation_output = dna_model.model.generate(batch['input_ids'].to(device),
                                                     min_length=dna_model.decoder_max_length-1,
                                                     max_length=dna_model.decoder_max_length-1,
                                                     bad_words_ids=bad_words_id,
                                                     num_beams=num_beams,
                                                     num_return_sequences=num_beams,
                                                     temperature=1,
                                                     num_beam_groups=1)

        prediction = dna_model.tokenizer.batch_decode(generation_output,skip_special_tokens=True)
        predictions += prediction 

    filtered_predictions = []
    chunks_of_predictions = list2splits([seq.replace(" ","") for seq in predictions],num_beams)
    
    input_strands = list(error_df["dna_copy"])

    for prediction_chunk,input_str in zip(chunks_of_predictions,input_strands):
        dist_list = []
        for pred in prediction_chunk:
            edit_dist = edist.sed.sed_string(pred, input_str)
            if edit_dist==0:
                edit_dist = 100
            dist_list.append(edit_dist)

        index, _ = min(enumerate(dist_list), key=operator.itemgetter(1))

        filtered_predictions.append(prediction_chunk[index])

    pred_str = filtered_predictions 
    label_str = list(error_df["orig_DNA"])
    pred_hamming_distance = batch_normalized_hamming_distance(pred_str,label_str,return_list=True)
    subs,dels,ins,error_rate = edit_distance_summ(label_str,pred_str)

    error_df['dna_recoverd'] = pred_str
    error_df['pred_hamming_distance'] = pred_hamming_distance
    error_df['pred_is_error'] = (error_df['dna_recoverd']!=error_df["orig_DNA"]).astype(int)

    HammingScore = error_df['pred_hamming_distance'].mean()
    ErrorScore = error_df['pred_is_error'].mean()
    AccuracyScore = 1-ErrorScore

    print(f"HammingScore: {HammingScore*100} %   Edit_error_rate:{error_rate*100} %")
    metrics = {"HammingScore":HammingScore,
                "ErrorScore":ErrorScore,
                "AccuracyScore":AccuracyScore,
                "Edit_error_rate":error_rate,
                "Substitution_rate:":subs,
                "Deletion_rate:":dels,
                "Insertion_rate:":ins,
              }
    torch.cuda.empty_cache()
    del dna_model
    return error_df,metrics

def PredictEvalAll(predictions_sets,model_dir=None,batch_size=128,error_codewords=None,selective_decoding=False,GTGS_As_CL=False):
                    """
    *Predicts and decodes error strings in a data frame*
    
        parameters:
        ----------
            :param predictions_sets:
                type: dict
                A dictionary with the data divided to sets by length and correctness of codewords
            :param model_dir:
                type: str 
                the directory of the model
            :param error_codewords:
                type: list
                Default: None
                A list with the error codewords
            :param batch_size:
                type: int 
                Default: 128
                The batch size for prediction
            :param bselective_decoding:
                type: boolean
                Default: False
                Apply selective decoding or not
            :param GTGS_As_CL:
                type: boolean
                Default: False
                If true, it means there are no unrealible codewords so the GTGS is treated as CL


                

        returns:
        --------
       The df with the predictions and the all the errors metrices

    """
    if model_dir:
        models_perfix = model_dir
    else:
        models_perfix = f"model/{data_type}/best_model_4_mer_"
    metrics = {}
    df_preds = {}
    for idx in predictions_sets.keys():
        print("task: ",idx)
        df = predictions_sets[idx]

        if idx == "GTGS":
            if GTGS_As_CL:
                model_name = models_perfix+"CL"
                print("Treating GTGS as CL...")
                print("model_name: ",model_name)
                if selective_decoding:
                    df,metric = SelectivePredictEndDecode(df,model_name,error_codewords,batch_size=batch_size)
                else:
                    df,metric = PredictEndDecode(df,model_name,error_codewords=None,batch_size=batch_size)
            else:
                print("model_name: ","NO MODEL! GTGS!")
                df,metric = MakeGTGSResults(df)
        else:
            model_name = models_perfix+idx
            print("model_name: ",model_name)
            if selective_decoding:
                df,metric = SelectivePredictEndDecode(df,model_name,error_codewords,batch_size=batch_size)
            else:
                df,metric = PredictEndDecode(df,model_name,error_codewords=None,batch_size=batch_size)

        df["split_type"] = idx
        metrics[idx] = metric
        df_preds[idx] = df

    prediction_df = pd.concat([df_preds["SL"],df_preds["LL"],df_preds["CL"],df_preds["GTGS"]])
    
    subs,dels,ins,error_rate = edit_distance_summ(list(prediction_df['orig_DNA']),list(prediction_df['dna_recoverd']))
    HammingScore = prediction_df['pred_hamming_distance'].mean()
    ErrorScore = prediction_df['pred_is_error'].mean()
    AccuracyScore = 1-ErrorScore
    metrics["final"] = {"HammingScore":HammingScore,
                        "ErrorScore":ErrorScore,
                        "AccuracyScore":AccuracyScore,
                        "Edit_error_rate":error_rate,
                        "Substitution_rate:":subs,
                        "Deletion_rate:":dels,
                        "Insertion_rate:":ins,
                        }
    print("\n"*3,"-"*30,"FINAL RESULTS","-"*30,"\n"*3)
    print(f"HammingScore: {HammingScore*100} %")
    print(f"Edit_error_rate: {error_rate*100} %")
    print(f"ErrorScore: {ErrorScore*100} %")
    print(f"AccuracyScore: {AccuracyScore*100} %")
    return prediction_df,metrics