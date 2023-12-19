from sklearn.model_selection import train_test_split
import os
import datasets
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import random
from varname import argname2 
from .utils import batch_normalized_hamming_distance,kmer2seq,seq2kmer,seq2splits
from .configuration import read_config

configs = read_config()

output_loc = configs["output_loc"]
dataset_dir = f"{output_loc}/datasets"
PROJECT_FOLDER = configs["PROJECT_FOLDER"]
random_state = configs["random_state"]
k_mer_size = configs["k_mer_size"]
data_type = configs["data_type"]


def TrainTestSplitByDocs(error_df,eval_and_test=True,
                       eval_ratio=0.15,test_ratio=0.2,
                       doc_column_name="doc_id",
                       random_state=random_state,
                       HammingScore_column="HammingScore",
                       do_print=True):
    """
    *Splits data frame to train and test* 
    
        parameters:
        ----------
            :param error_df:
                type: pd.data frame
                the data frame with errored strings
            :param eval_and_test:
                type: boolean
                Default: True
                Wether you want eval and test or just eval
            :param eval_ratio:
                type: float between 0 and 1
                Default: 0.15
                The ratio of eval set
            :param test_ratio:
                type: float between 0 and 1
                Default: 0.2
                The ratio of test set
            :param edoc_column_name:
                type: str
                Default: "doc_id"
                The column according to which the split is made
            :param HammingScore_column:
                type: str
                Default: "HammingScore"
                The column with hamming score
            :param do_print:
                type: boolean
                Default: True
                If  true, print info about the split

        returns:
        --------
       A list with the train_df,eval_dt,text_df 
        """
    data_names = ['train_df','eval_df','test_df']
    doc_ids = list(error_df[doc_column_name].unique())
    
    if eval_and_test:
        train_eval_ratio = eval_ratio+test_ratio
        test_eval_ratio = (test_ratio+eval_ratio)/2
        #test_eval_ratio = test_ratio


        train_docs,test_eval_docs = train_test_split(doc_ids,test_size=train_eval_ratio, random_state=random_state)
        eval_docs,test_docs = train_test_split(test_eval_docs,test_size=test_eval_ratio, random_state=random_state)
        train_df = error_df.loc[error_df[doc_column_name].isin(train_docs)]
        eval_df = error_df.loc[error_df[doc_column_name].isin(eval_docs)]
        test_df = error_df.loc[error_df[doc_column_name].isin(test_docs)]
        data_splits = [train_df,eval_df,test_df]

    else:
        train_docs,eval_docs = train_test_split(doc_ids,test_size=eval_ratio, random_state=random_state)
        train_df = error_df.loc[error_df[doc_column_name].isin(train_docs)]
        eval_df = error_df.loc[error_df[doc_column_name].isin(eval_docs)]
        data_splits = [train_df,eval_df]
    if do_print:  
        print("Length:")
        print(f"Full data: {len(error_df)}",[f"{data_names[i]}: {len(data_splits[i])}" for i in range(len(data_splits))])
        if HammingScore_column:
            print("Mean HammingScore:")
            print(f"Full data: {error_df[HammingScore_column].mean()}",[f"{data_names[i]}: {round(data_splits[i][HammingScore_column].mean(),6)}" for i in range(len(data_splits))])
    return data_splits

def LoadDataset(file_path):
    """
    *load data frame from a csv*
    """
    dataset = load_dataset('csv', data_files=file_path)['train']
    return dataset

def Data2Dataset(*dfs,errored_column_name="dna_copy",label_column_name="orig_DNA",make_pre_train=False):
    """
    *Create a dataset for training* 
    
        parameters:
        ----------
            :param *dfs:
                type: list
                list of all dataframes we wish to use for the dataset
            :param errored_column_name:
                type: str
                Default: "dna_copy"
                name of column with errored strings
            :param label_column_name:
                type: str
                Default: "orig_DNA"
                name of column with original strings
            :param make_pre_train:
                type: boolean
                Default: False:
                If  true, print info about the split

        returns:
        --------
      Data frame of all the data
        """

    names = argname2('*dfs')
    for name, df in zip(names, dfs):
        title = name
        error_df = df
        
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    if make_pre_train:
        errored_column_name = label_column_name
    error_df = error_df.copy()
    error_df["kmer_W_errors"] = [seq2kmer(seq,4) for seq in error_df[errored_column_name]]
    error_df["split_orig"] = [seq2splits(seq,4) for seq in error_df[label_column_name]]
    
    relevant_cols = ['kmer_W_errors', 'split_orig','orig_id']
    file_path = f"{dataset_dir}/{title}.csv"
    error_df[relevant_cols].to_csv(file_path,index=False)  
    dataset = LoadDataset(file_path)
    return dataset

