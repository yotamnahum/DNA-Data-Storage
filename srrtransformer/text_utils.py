import os
import pandas as pd
from tqdm import tqdm
import re
import pickle
import glob
import itertools
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import random
import os 

from .utils import batch_normalized_hamming_distance,kmer2seq,seq2kmer,seq2splits
from .configuration import read_config

config = read_config()

output_loc = config["output_loc"]
dataset_dir = f"{output_loc}/datasets"
PROJECT_FOLDER = config["PROJECT_FOLDER"]
random_state = config["random_state"]
k_mer_size = config["k_mer_size"]
data_type = config["data_type"]
seq_length = config["seq_length"]

if data_type != 'text':
    print("WARNING! 'datatype' != 'text'")
with open(f'{PROJECT_FOLDER}/srrtransformer/dict/codewords_utf.pickle','rb') as f:
    translation_df = pickle.load(f)

DNA2Char_dict = {}
Char2DNA_dict = {}
codewords = translation_df['codeword'].to_list()
tokens = translation_df['token'].to_list()
for token,codeword in zip(tokens,codewords):
    DNA2Char_dict.update({codeword:token})
    Char2DNA_dict.update({token:codeword})


def _sub_tokenize(sub_text):
    """
    *Convert text to tokens*
            :param sub_text:
                type: str
                string representing text
                
                
    Returns: list of tokenized text (strings)   
          
    
    
    """
    character_list = list(sub_text)
    utf_tokens_lists = [list(char.encode("utf-8")) for char in character_list]
    sub_tokens = [chr(utf_token) for utf_tokens in utf_tokens_lists for utf_token in utf_tokens]
    return sub_tokens

def _convert_sub_string(sub_chars):
    """
    *Convert text to tokens*
            :param sub_chars:
                type: text
                string representing text
                
                
     Returns: list of tokenized text (strings)   
          
    
    
    """
    byte_string = bytes([ord(char) for char in sub_chars])
    return byte_string.decode("utf-8", errors="ignore")

def convert_tokens_to_string(tokens):
      """*Converts a sequence of tokens (string) in a single string.*
        
            :param tokens:
                type: str
                
            Returns: single string of all tokens     
        
        """
    string = ""
    sub_chars = []
    for token in tokens:
      # if is special token
        if len(token) > 1:
            string += _convert_sub_string(sub_chars)
            string += token
            sub_chars = []
        else:
            sub_chars.append(token)

    # add remaining chars
    string += _convert_sub_string(sub_chars)

    return string

def Char2DNA(text):
    """
    *Convert full text to DNA tokens*
            :param text:
                type: str
                string representing text
                
                
    Returns: list of tokenized text (strings)   
          
    
    
    """
    sub_tokens = _sub_tokenize(text)
    return [Char2DNA_dict[token] for token in sub_tokens]

def DNA2Char(kmers):
    """
    *Converts DNA strings(tokens) to text*
            :param kmers:
                type: list
                list with dna strings (tokens)
                
                
    Returns: list of original text strings   
          
    
    
    """
    tokens = [DNA2Char_dict[kmer] for kmer in kmers]
    recovered_string = convert_tokens_to_string(tokens)
    return recovered_string

def CheckBinaryDNAConversion(text,uncase=False):
    """
    *Converts a text to DNA and then to binary*
            :param text:
                type: str
                str representing text
                
                
    Returns: binary string  
          
    
    
    """
    if uncase:
        original_str = text.lower()
    else:
        original_str = text
    # convert text to binary values
    binary_str = ''.join(format(x, '08b') for x in bytearray(original_str, 'utf-8'))
    binary_list = [binary_str[i: i+2] for i in range(0, len(binary_str), 2)]

    # binary values to nucleotide sequence
    DNA_encoding = {
      "00": "A",
      "01": "G",
      "10": "C",
      "11": "T"
    } 

    DNA_list = []
    for num in binary_list:
        for key in list(DNA_encoding.keys()):
            if num == key:
                DNA_list.append(DNA_encoding.get(key))
    Text2DNA_binary = "".join(DNA_list)
    return Text2DNA_binary


def Chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def File2DNA(file_name=None, folder_name=None, pad_short_files=True,pad_marker_letter=" ",limit_doc_num=None):
                """
    *Create a data frame of the DNA encoding of an text or a set of texts* 
    
        parameters:
        ----------
            :param file_name:
                type: str.
                default: None.
                Location of the text file you wish to convert
            :param folder_name:
                type: str.
                default: None.
                Location of the folder of texts you wish to convert  
            :param pad_short_files:
                type: boolean
                default: True
                Pad short files to a fixed length
            :param pad_marker_lette:
                type: str
                default: " "
                pard the marker letter
            :param limit_doc_num:
                type: int.
                default: None.
                limit the number of images you wish to encode 

        returns:
        --------
        A data frame with fields:
        "DNA": DNA string
        "DNA_len": Length of the DNA string
        "text": the text that was encoded
        "text_len" length of the encdoed text
        "doc_id": id of image the DNA string came from
        "chunk_id":id of squared block the DNA string represents
        "is_padded": whether it was padded or not
        "lang": lang
        
        """
    """"return a df of chunks from single .txt file OR multiple .txt files"""
    if file_name and folder_name:
        print("Please choose file_name (for single file) OR folder_name (for multiple files)!")
    if file_name:
        txt_files = [file_name]
    if folder_name:
        txt_files = glob.glob(f"{folder_name}/*.txt")
        if limit_doc_num:
            txt_files = txt_files[:limit_doc_num]

    pad_marker = Char2DNA(pad_marker_letter)[0]
    chunk_size = int(seq_length/k_mer_size)
    doc_list = []
    doc_id = 0
    for file in tqdm(txt_files):
        with open(file, 'r') as fd:
            doc = fd.read()
            lang = file.split("/")[0]
            trans_doc = Char2DNA(doc)
            chunked_doc = list(Chunks(trans_doc,chunk_size))
            chunk_id = 0
            for chunk in chunked_doc:
                is_padded = 0
                DNA_str = "".join(chunk)
                DNA_len = len(DNA_str)
                text = DNA2Char(chunk)
                text_len = len(text)
                if pad_short_files:
                    if DNA_len<seq_length:
                        len_diff = int((seq_length - DNA_len)/k_mer_size)
                        DNA_str = DNA_str+(pad_marker*len_diff)
                        DNA_len = len(DNA_str)
                        is_padded = 1
                doc_list.append({"DNA":DNA_str,"DNA_len":DNA_len,"text":text,"text_len":text_len,"doc_id":doc_id,"chunk_id":chunk_id,"is_padded":is_padded,"lang":lang})
                chunk_id+=1
        doc_id+=1
    df = pd.DataFrame(doc_list)
    df["orig_id"] = range(len(df))
    return df
