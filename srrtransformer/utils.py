import os
import statistics
import pickle
import pandas as pd
import numpy as np
import textdistance
from .configuration import read_config
import sys

config = read_config()
dnaSimulator_dir = "external_utils/dnaSimulator"
sys.path.insert(1,dnaSimulator_dir)
normalized_hamming_distance = textdistance.hamming.normalized_distance

def batch_normalized_hamming_distance(pred_str,label_str,return_list=False):
                    """
    *Simulated errors on DNA string\DF of dna strings*
    
        parameters:
        ----------
            :param pred_str:
                type: list
               list of predicted strings
            :param label_str:
                type: list
               list of original strings
            :param return_list:
                type: boolean
                Default: False
                If true, return a list of hamming distance of every (pred,label) pair, otherwise, return the mean hamming distance

        returns:
        --------        
        If true, return a list of hamming distance of every (pred,label) pair, otherwise, return the mean hamming distance
    """
    HammingScore_list = []
    for pred,label in zip(pred_str,label_str):
        pred = pred.replace(" ","")
        label = label.replace(" ","")
        HammingScore_list.append(round(normalized_hamming_distance(pred,label),6))
        
        if return_list:
            output = HammingScore_list
        else:
            output = np.mean(HammingScore_list)
      
    return output


def kmer2seq(kmers):
    """
    *Convert kmers to original sequence Arguments*
    
            :param kmers:
                type: str
                kmers separated by space

    Returns:
    seq -- str, original sequence.
    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    Arguments:
            :param seq:
                type: str
                original sequence
            :param k:
                type: int
                kmer of length k specified

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


def seq2splits(seq, k):
    """
    Convert original sequence to chunks in size k, splited by " "
    Arguments:
            :param seq:
                type: str
                original sequence
            :param k:
                type: int
                kmer of length k specified

    Returns:
    splits by " " -- str, kmers separated by space
    """
    chunks = len(seq)
    chunk_size = k
    split_seq = " ".join([seq[i:i+chunk_size] for i in range(0,chunks,chunk_size)])
    return split_seq

def SimulateErrors(DNA_data,num_copies=2,DNA_column="DNA",error_factor=1): 
                    """
    *Simulated errors on DNA string\DF of dna strings*
    
        parameters:
        ----------
            :param DNA_data:
                type: pd.dataframe or str
                A DNA string or a dataframe of DNA strings
            :param num_copies:
                type: int 
                Default: 2
                dectated how many error copies are produced for each string
            :param DNA_column:
                type: str
                Default: "DNA"
                The column which contains the DNA strings
            :param error_factor:
                type: float 
                Default: 1
                The error probabilites are multiplied by this factor
                

        returns:
        --------        
        A data frame with fields:
        "orig_DNA": original DNA string
        "dna_copy": error DNA string
        "orig_DNA_len": Length of the  original DNA string
        "dna_copy_len": Length of the DNA copy       
        "orig_id": id of the original DNA string 
        "copy_id": id of copy DNA
        "HammingScore": Hamming score

    """
    DNA_data_type = type(DNA_data)
    if DNA_data_type is pd.core.frame.DataFrame:
        list_of_DNA = list(DNA_data[DNA_column])
    elif DNA_data_type is str:
        list_of_DNA = [DNA_data]
    else:
        list_of_DNA = list(DNA_data)
        
    from simulator import Simulator
    error_rates_example = {'d':9.58 * (10 ** (-4))*error_factor,
                          'ld': 2.33 * (10 ** (-4))*error_factor,
                          'i': 5.81 * (10 ** (-4))*error_factor,
                          's': 1.32 * (10 ** (-3))*error_factor}
    base_error_rates_example = {'A':
                                {'s': 0.135 * (10**(-2))*error_factor,
                                'i': 0.057 * (10**(-2))*error_factor,
                                'pi': 0.059 * (10**(-2))*error_factor,
                                'd': 0.099 * (10**(-2))*error_factor,
                                'ld': 0.024 * (10**(-2))*error_factor},
                                'C':
                                    {'s': 0.135 * (10 ** (-2))*error_factor,
                                    'i': 0.059 * (10 ** (-2))*error_factor,
                                    'pi': 0.058 * (10 ** (-2))*error_factor,
                                    'd': 0.098 * (10 ** (-2))*error_factor,
                                    'ld': 0.023 * (10 ** (-2))*error_factor},
                                'T':
                                    {'s': 0.126 * (10 ** (-2))*error_factor,
                                    'i': 0.059 * (10 ** (-2))*error_factor,
                                    'pi': 0.057 * (10 ** (-2))*error_factor,
                                    'd': 0.094 * (10 ** (-2))*error_factor,
                                    'ld': 0.023 * (10 ** (-2))*error_factor},
                                'G':
                                    {'s': 0.132 * (10 ** (-2))*error_factor,
                                    'i': 0.058 * (10 ** (-2))*error_factor,
                                    'pi': 0.058 * (10 ** (-2))*error_factor,
                                    'd': 0.096 * (10 ** (-2))*error_factor,
                                    'ld': 0.023 * (10 ** (-2))*error_factor}}


    DNA_strings = "\n".join(list_of_DNA)
    filename = "dna_input.txt"
    with open(filename, 'w') as out:
        out.write(DNA_strings)

    distribution_example = {'type': 'vector', 'value': [num_copies]*len(list_of_DNA)}
    sim = Simulator(error_rates_example, base_error_rates_example, filename, False, distribution_example)
    sim.simulate_errors(report_func=None)

    error_data = []

    string_id = 0
    with open("output/evyat.txt", 'r') as out:
        output = out.read()
    for chunk in output.split("\n\n\n")[:-1]:
        splited_chunk = chunk.split("\n*****************************\n")
        orig_DNA = splited_chunk[0]
        orig_DNA_len = len(orig_DNA)
        orig_id = string_id
        copy_id = 0
        dna_copies = splited_chunk[1].split("\n")
        for dna_copy in dna_copies:
            dna_copy_len = len(dna_copy)
            is_error = int(orig_DNA!=dna_copy)
            if is_error==1 and len(orig_DNA)==len(dna_copy):
                is_subst = 1
            else:
                is_subst = 0
            error_data.append({"orig_DNA":orig_DNA,"dna_copy":dna_copy,"is_error":is_error,"is_subst":is_subst,"orig_DNA_len":orig_DNA_len,"dna_copy_len":dna_copy_len,"orig_id":orig_id,"copy_id":copy_id})
            copy_id+=1
            
        string_id+=1
    dna_df = pd.DataFrame(error_data)
    dna_df['HammingScore'] = batch_normalized_hamming_distance(list(dna_df["dna_copy"]),list(dna_df["orig_DNA"]),return_list=True)
    if DNA_data_type is pd.core.frame.DataFrame:
        #DNA_data.drop("DNA",axis=1,inplace=True)
        dna_df = dna_df.merge(DNA_data,how='left',on='orig_id').copy()

    return dna_df