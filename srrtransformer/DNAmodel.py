import os
import datasets
import transformers
from datasets import load_dataset
from transformers import BertTokenizerFast, EncoderDecoderModel,EncoderDecoderConfig,Seq2SeqTrainingArguments,Seq2SeqTrainer
import statistics
import pickle
import numpy as np
import textdistance
import torch
import edist.sed
from .utils import batch_normalized_hamming_distance,kmer2seq,seq2kmer,seq2splits
from .configuration import read_config
from .data_utils import Data2Dataset

if torch.cuda.device_count() > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
rouge = datasets.load_metric("rouge")   
config = read_config()
pretrained_model_path = config["pretrained_model_path"]


def detailed_edit_distance(string_b,string_a):
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
        if align[len(align)-1] == "-":
            num_of_deletions=num_of_deletions+1
        if align[0] == "-":
            num_of_insertions = num_of_insertions+1
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
    
class DnaModel:
    
    def __init__(self, batch_size=32,from_pretrained=False,pretrained_model_path=None,model_dir=None):
        """
        *Build a DNA model based on the DNA Encoder-Decoder model *
        parameters:
        ----------
            :param batch_size:
                type: int
                batch size for processing of the data
            :param from_pretrained:
                type: boolean.
                default: True. 
                Innitiate the model with pretrained weights or not
            :param model_dir:
                type: str. 
                default: None. 
                Directory of the model
    
            """
        self.batch_size = batch_size
        self.from_pretrained = from_pretrained

        if pretrained_model_path:
            self.model_dir = config["pretrained_model_path"]
        elif model_dir:
            self.model_dir = model_dir
        else:
            self.model_dir = "models/encoder_decoder_config"
        orig_sequance_length = config["seq_length"]
        self.kmer_size = config["k_mer_size"]
        kmer_size = self.kmer_size
        self.special_token_number = 2
        
        self.orig_sequance_kmer_length = int(orig_sequance_length/kmer_size)
        safe_factor = 4
        self.encoder_max_length = self.kmer_size*self.orig_sequance_kmer_length - (self.kmer_size-1) + safe_factor
        self.decoder_max_length = self.orig_sequance_kmer_length + self.special_token_number

        tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        self.tokenizer = tokenizer

    def process_data_to_model_inputs(self,batch):
                    """
        *Process the data to prepare for model training*
        ----------
            :param batch:
                type: dictionary
                dictionary with DNA strings data
         Returns:
         dictionary with the processed data
            """
        tokenizer = self.tokenizer
        # tokenize the inputs and labels
        inputs = tokenizer(batch["kmer_W_errors"], padding="max_length", truncation=True, max_length=self.encoder_max_length)#self.encoder_max_length)
        outputs = tokenizer(batch["split_orig"], padding="max_length", truncation=True, max_length=self.decoder_max_length)#self.decoder_max_length)
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
        return batch

    def format_data(self,data):
                    """
        *set format of the data to prepare for input to the model*
        ----------
            :param data:
                type: pd.dataframe
                df with the DNA strings data 
         Returns:
         df with the processed and formated data
            """
        data = data.map(
            self.process_data_to_model_inputs, 
            batched=True, 
            batch_size=self.batch_size)
        data.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])
        return data

    def load_model(self):
                         """
        *load the model*
   
            """
        if self.from_pretrained:
            bert2bert = EncoderDecoderModel.from_pretrained(self.model_dir)
        else:
            bert2bert_config = EncoderDecoderConfig.from_json_file("models/encoder_decoder_config/config.json")
            bert2bert = EncoderDecoderModel(config=bert2bert_config)

        bert2bert.config.decoder_start_token_id = self.tokenizer.bos_token_id
        bert2bert.config.eos_token_id = self.tokenizer.eos_token_id
        bert2bert.config.pad_token_id = self.tokenizer.pad_token_id
        # sensible parameters for beam search
        bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
        bert2bert.config.max_length = self.decoder_max_length - 1
        bert2bert.config.min_length = self.decoder_max_length - 1
        bert2bert.config.no_repeat_ngram_size = 0
        bert2bert.config.early_stopping = True
        bert2bert.config.length_penalty = 1.0
        bert2bert.config.num_beams = 4
        bert2bert.to(device)
        self.model = bert2bert
    
    def compute_metrics(self,pred):
        tokenizer = self.tokenizer
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
        label_avg_len = []
        pred_avg_len = []
        accuracy_list = []
        joined_label = []
        joined_pred = []

        for label,pred in zip(label_str,pred_str):
            label_avg_len.append(len(label.split(" ")))
            pred_avg_len.append(len(pred.split(" ")))
            accuracy_list.append(int(label==pred))
            joined_label.append(label.replace(" ",""))
            joined_pred.append(pred.replace(" ",""))
            
        subs,dels,ins,error_rate = edit_distance_summ(joined_label,joined_pred)
        
        HammingScore = batch_normalized_hamming_distance(joined_pred,joined_label)
        
        accuracy = np.mean(accuracy_list)

        print("label_str: ",label_str[:5])
        print("label_avg_len: ",statistics.mean(label_avg_len))
        print("pred_str: ",pred_str[:5])
        print("pred_avg_len: ",statistics.mean(pred_avg_len))

        {"Error rate":error_rate,
         "Substitution rate:":subs,
         "Deletion rate:":dels,
         "Insertion rate:":ins}
        
        return {
            "NEG_HammingScore": 1-round(HammingScore, 4),
            "HammingScore": round(HammingScore, 4),
            "Edit_error_rate":error_rate,
            "Substitution_rate:":subs,
            "Deletion_rate:":dels,
            "Insertion_rate:":ins,
            "Accuracy": round(accuracy, 4),
            "Error_score": 1-round(accuracy, 4),
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }
        
    def train_reconstruction_model(self,train_df,eval_df):
        train_data = format_data(Data2Dataset(train_df))
        eval_data = format_data(Data2Dataset(eval_df))
        output_loc = config["output_loc"]
        total_steps_for_epoch = int(len(train_data)/self.batch_size)
        eval_steps = int(total_steps_for_epoch/1)
        save_steps = int(eval_steps*3)
        logging_steps = 50 

        print(f"total_steps_for_epoch: {total_steps_for_epoch}")
        print(f"eval_steps: {eval_steps}")
        print(f"save_steps: {save_steps}")
        print(f"logging_steps: {logging_steps}")
        training_args = Seq2SeqTrainingArguments(
        learning_rate=4e-5,
        output_dir=output_loc,
        evaluation_strategy="steps",
        num_train_epochs = 15,
        per_device_train_batch_size=self.batch_size,
        per_device_eval_batch_size=self.batch_size,
        predict_with_generate=True,
        metric_for_best_model = "Edit_error_rate",
        greater_is_better = True,
        load_best_model_at_end = True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        warmup_steps=300,
        overwrite_output_dir=True,
        save_total_limit=10,
        fp16=True,
        run_name=run_name,
        weight_decay=0.01,
        )

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=eval_data,
        )
        return trainer    

