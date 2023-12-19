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
from skimage.util.shape import view_as_blocks
import random
import os 
from PIL import Image
from .utils import batch_normalized_hamming_distance,kmer2seq,seq2kmer,seq2splits
from .configuration import read_config

config = read_config()

output_loc = config["output_loc"]
dataset_dir = f"{output_loc}/datasets"
PROJECT_FOLDER = config["PROJECT_FOLDER"]
random_state = config["random_state"]
k_mer_size = config["k_mer_size"]
data_type = config["data_type"]
patch_size = tuple(config["patch_size"])
Normalized = True



if data_type != 'image':
    print("WARNING! 'datatype' != 'image'")
with open(f'{PROJECT_FOLDER}/srrtransformer/dict/codewords_pixel.pickle','rb') as f:
    translation_df = pickle.load(f)
    codewords = translation_df['codeword'].to_list()
    tokens = translation_df['token'].to_list()
DNA_dictionary = {}
for codeword,token in zip(codewords,tokens):
    DNA_dictionary[token] = codeword
key_list = list(DNA_dictionary.keys())
val_list = list(DNA_dictionary.values())



def convert_gray_image_to_dna_diag_normalized(image,dict):
    """
    *Convert a gray squared image to a list in which each instance is the DNA encoding value of each pixel according to a dictionary
    if Normalized == True (in configs, see above), Each value is calculated with respect to the minimum value of the image (subtruction), 
    and the first item in the list is the minimum value encoded with dna.
    The word diag in the definition refers to the fact that pixels are saved according to the order of the diagonals rather then the lines of the image* 
    
        parameters:
        ----------
            :param image:
                type: np.ndarray
                A two dimensional squared numpy array of integers that appear in the key list of dict
            :param dict:
                type: dictionary
                the keys are numbers between 0-255 and the values are strings over {A,T,C,G} of a fixed size

        returns:
        --------
        A list in which each instance is the DNA encoding value of each pixel (strings over {A,T,C,G})
        the length of the list is the size of the image
        """
    if Normalized==True:
        answer = []
        min_value = np.min(image)
        answer.append(dict[min_value])
        size = image.shape[0],image.shape[1]
        diags = [image[::-1,:].diagonal(i) for i in range(-image.shape[0],image.shape[1])]
        for d in range(len(diags)):
            if d%2==0:
                arr=diags[d]
                for j in range(len(diags[d])):
                    answer.append(dict[arr[j]-min_value])
            if d%2==1:
                arr = diags[d][::-1]
                for j in range(len(diags[d])):
                    answer.append(dict[(arr[j]-min_value)])
                    
    if Normalized==False:
        answer = []

        size = image.shape[0],image.shape[1]
        diags = [image[::-1,:].diagonal(i) for i in range(-image.shape[0],image.shape[1])]
        for d in range(len(diags)):
            if d%2==0:
                arr=diags[d]
                for j in range(len(diags[d])):
                    answer.append(dict[arr[j]])
            if d%2==1:
                arr = diags[d][::-1]
                for j in range(len(diags[d])):
                    answer.append(dict[(arr[j])])
    return answer

def convert_dna_to_gray_image_diag_normalized(Dna_string):
        """
    *Converts a list of DNA strings to a gray squared image (According to DNA_dictionary configured at the beginning of this file)
    if Normalized == True (in configs, see above), Each value in the list represents the absolut value of the pixel with respect to the minimum value of the image (subtruction), and the first item in the list is the minimum value encoded with dna* 
    
        parameters:
        ----------
            :param dna_list:
                type: List
                 Each instance in the list is a string over {A,T,C,G} that appears in the DNA_dictionary value list

        returns:
        --------
        A two dimenstional squared numpy array of inegeres that appear in the key list of DNA_dictionary
        """
    if Normalized==True:
        answer = np.ndarray(shape = patch_size,dtype = int)
        min_value = val_list.index(Dna_string[0])
        Dna_string = Dna_string[1:]
        sum = 0 
        String_size = patch_size[0]-1 
        Total_size = len(Dna_string)-1 
        for d in range(patch_size[0]): 
            sum = sum+d
            if d%2==0: 
                j = d
                for i in range(d+1):
                    answer[i,j] = val_list.index(Dna_string[sum+i])+min_value 
                    answer[String_size-i,String_size-j] = val_list.index(Dna_string[Total_size - sum - i])+min_value 
                    if j==0:
                        break
                    j-=1
            if d%2==1: 
                i = d
                for j in range(d+1):
                    answer[i,j] = val_list.index(Dna_string[sum+j])+min_value
                    answer[String_size-i,String_size-j] = val_list.index(Dna_string[Total_size - sum - j])+min_value
                    if i==0:
                        break
                    i-=1
                    
    if Normalized==False:
        answer = np.ndarray(shape = patch_size,dtype = int)
 
        sum = 0 
        String_size = patch_size[0]-1 
        Total_size = len(Dna_string)-1 
        for d in range(patch_size[0]): 
            sum = sum+d
            if d%2==0: 
                j = d
                for i in range(d+1):
                    answer[i,j] = val_list.index(Dna_string[sum+i])
                    answer[String_size-i,String_size-j] = val_list.index(Dna_string[Total_size - sum - i]) 
                    if j==0:
                        break
                    j-=1
            if d%2==1: 
                i = d
                for j in range(d+1):
                    answer[i,j] = val_list.index(Dna_string[sum+j])
                    answer[String_size-i,String_size-j] = val_list.index(Dna_string[Total_size - sum - j])
                    if i==0:
                        break
                    i-=1
    return answer

def convert_block_matrix_to_dna_string(B_matrix):
            """
    *Converts a Block matrix to a list in which every instance is  a list representing the DNA encoding the each block of the matrix* 
    
        parameters:
        ----------
            :param B_matrix:
                type: np.ndarray
                 4-dimensionaly numpy array represents a block matrix. first two dimensions are the shape of the block matrix, and the former two are 
                 the size of each block.

        returns:
        --------
        A list in which every instance is  a list representing the DNA encoding the each block of the matrix
        """
    dna_string = []
    matrix_shape = B_matrix.shape
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            dna_patch = convert_gray_image_to_dna_diag_normalized(B_matrix[i,j],DNA_dictionary)
            dna_string.append(dna_patch)
    return dna_string    

def File2DNA(file_name=None, folder_name=None,suffix=None,limit_doc_num=None):
                """
    *Create a data frame of the DNA encoding of an image or a set of images* 
    
        parameters:
        ----------
            :param file_name:
                type: str.
                default: None.
                Location of the image file you wish to convert
                image dimensions must be divisible in 5
            :param folder_name:
                type: str.
                default: None.
                Location of the folder of images you wish to convert
                image dimensions must be divisible in 5    
            :param suffix:
                type: str.
                default: None.
                type of image\s you wish to encode e.g png\jpeg. if None, every type of image will be encoded
            :param limit_doc_num:
                type: int.
                default: None.
                limit the number of images you wish to encode 

        returns:
        --------
        A data frame with fields:
        "DNA": DNA string
        "DNA_len": Length of the DNA string
        "doc_id": id of image the DNA string came from
        "color_id": id of color channel the DNA string came from: 1:Red 2:Green 3:Blue
        "chunk_id":id of squared block the DNA string represents
        "DNA_min_value": minimum value of the squared block
        "orig_image_size": original image dimensions
        
        """
    if file_name and folder_name:
        print("Please choose file_name (for single file) OR folder_name (for multiple files)!")
    if file_name:
        image_address_list = [file_name]
    if folder_name:
        directory = folder_name
        image_address_list = glob.glob(f"{directory}/*{suffix}")
        if limit_doc_num:
            image_address_list = image_address_list[:limit_doc_num]
            
    path_list = []
    byte_list = []
    size_list = []
    image_list = []

    for img in tqdm(image_address_list):
        temp_image = Image.open(img)
        to_keep = temp_image.copy()
        image_list.append(to_keep)
        temp_image.close()

    np_one_channel_image_list = []
    for img in tqdm(image_list):
        np_img = np.array(img)

        print(np_img.shape[2])
        if (np_img.shape[2]==3) & (np_img.shape[0]%5==0) & (np_img.shape[1]%5==0):
            red_img = np_img[:,:,0]
            green_img = np_img[:,:,1]
            blue_img = np_img[:,:,2]
            np_one_channel_image_list.append(red_img)
            size_list.append(red_img.shape)
            np_one_channel_image_list.append(green_img)
            size_list.append(green_img.shape)
            np_one_channel_image_list.append(blue_img)
            size_list.append(blue_img.shape)
    print("There are " + str(len(np_one_channel_image_list)/3) + " images in the dataset" )

    Block_matrix_list = [] 

    for img in tqdm(np_one_channel_image_list):
        Block_matrix_img = view_as_blocks(img, block_shape=patch_size)
        Block_matrix_list.append(Block_matrix_img)

    one_channel_image_dna_list=[] 

    for i in tqdm(range(len(Block_matrix_list))):
        B_matrix = Block_matrix_list[i]
        img_as_dna_string = convert_block_matrix_to_dna_string(B_matrix)
        one_channel_image_dna_list.append(img_as_dna_string)

    doc_list = []
    doc_id = 0
    color_id = 0
    if Normalized==True:
        for i in tqdm(range(len(one_channel_image_dna_list))):
            img = one_channel_image_dna_list[i]
            image_size = size_list[i]
            chunk_id = 0

            for j in range(len(img)):
                DNA_min_value = img[j][0]
                DNA_str = img[j][1:]
                DNA_str = ''.join(map(str, DNA_str))
                DNA_len = len(DNA_str)
                doc_list.append({"DNA":DNA_str,"DNA_len":DNA_len,"doc_id":doc_id,"color_id":color_id,"chunk_id":chunk_id,"DNA_min_value":DNA_min_value,"orig_image_size": image_size})
                chunk_id+=1

            color_id+=1

            if color_id > 2:
                doc_id+=1
                color_id=0

        df = pd.DataFrame(doc_list)
        df["orig_id"] = range(len(df))  
    
    if Normalized==False:
        DNA_min_value = "-"
        for i in tqdm(range(len(one_channel_image_dna_list))):
            img = one_channel_image_dna_list[i]
            image_size = size_list[i]
            chunk_id = 0
            for j in range(len(img)):
                DNA_min_value = "dummy"
                DNA_str = img[j]
                DNA_str = ''.join(map(str, DNA_str))
                DNA_len = len(DNA_str)
                doc_list.append({"DNA":DNA_str,"DNA_len":DNA_len, "doc_id":doc_id,"color_id":color_id,"chunk_id":chunk_id,"DNA_min_value":DNA_min_value,"orig_image_size": image_size})
                chunk_id+=1

            color_id+=1

            if color_id > 2:
                doc_id+=1
                color_id=0

        df = pd.DataFrame(doc_list)
        #df["DNA_len"] = [len(dna) for dna in df["DNA"]]
        df["orig_id"] = range(len(df))
        
    return df

def convert_from_dna_to_block_matrix(dna_list,image_size,length_of_codewords=k_mer_size): 
        """
    *Create a block matrix from dna list* 
    
        parameters:
        ----------
            :param dna_lists:
                type: List.
                A list in which every instance is a list that represents the DNA encoding of a squared matrix
            :param image_size:
                type: tuple
                the dimensions of the image from which the dna list came from
            :param length_of_codewords:
                type: int
                default: k_mer_size(see configs above).
                the size of each value in the dictionary (size of DNA strings encoding the pixel levels) 

        returns:
        --------
        A block matrix of dimensions: image_size[0]/patch_size[0], image_size[1]/patch_size[1], patch_size[0],patch_size[1]
        where patch size is the size of each block (see configs above)
        """
    block_matrix_size=(int(image_size[0]/patch_size[0]),int(image_size[1]/patch_size[1]),int(patch_size[0]),int(patch_size[1]))
    answer = np.ndarray(shape = block_matrix_size,dtype = int)
    for i in range(block_matrix_size[0]):
        for j in range(block_matrix_size[1]):
            dna_list_to_convert = (dna_list[i*block_matrix_size[0]+j])
            dna_list_to_convert_split_to_n = [dna_list_to_convert[i:i+length_of_codewords] for i in range(0,len(dna_list_to_convert),length_of_codewords)]
            answer[i,j] = convert_dna_to_gray_image_diag_normalized(dna_list_to_convert_split_to_n)
    return answer

def convert_from_block_matrix_to_image(block_matrix,image_size):   
        """
    *Construct a gray image from block image* 
    
        parameters:
        ----------
            :param B_matrix:
                type: np.ndarray
                A 4 dimensional numpy array representing a block matrix
            :param image_size:
                type: tuple
                the dimensions of the image we wish to construct

        returns:
        --------
        A 2 dimensional np.ndarray with diemnsions: image_size
        
        """                                                                     
    original_image = block_matrix.transpose(0,2,1,3).reshape(image_size)
    return original_image

def ListSplits(seq):
        """
    *Splits a list to 3 lists * 
    
        parameters:
        ----------
            :param list:
                type: List.


        returns:
        --------
    A list of 3 lists
    """
    k = int(len(seq)/3)
    chunks = len(seq)
    chunk_size = k
    split_seq = [seq[i:i+chunk_size] for i in range(0,chunks,chunk_size)]
    return split_seq

def JoinChunnles(dna_list,image_size):
        """
    *Construct a color image from dna list* 
    
        parameters:
        ----------
            :param dna_list:
                type: List
                List fo dna strings
            :param image_size:
                type: tuple
                the dimensions of the image we wish to construct

        returns:
        --------
        np.ndarray of dimensions (image_size[0],image_size[1],3)        
        """
    image_shape=(image_size[0],image_size[1],3)
    image_matrix = np.ndarray(shape=image_shape,dtype=int).astype('uint8')
    im_array_R,im_array_G,im_array_B = ListSplits(dna_list) ###
    image_matrix[:,:,0] = convert_from_block_matrix_to_image(convert_from_dna_to_block_matrix(im_array_R,image_size),image_size)
    image_matrix[:,:,1] = convert_from_block_matrix_to_image(convert_from_dna_to_block_matrix(im_array_G,image_size),image_size)
    image_matrix[:,:,2] = convert_from_block_matrix_to_image(convert_from_dna_to_block_matrix(im_array_B,image_size),image_size)
    return image_matrix

def Str2Tuple(size_string):
   """
    *Converts a list to a tuple* 
    
        parameters:
        ----------
            :param size_string:
                type: List
            :param image_size:
        returns:
        --------
        tuple       
        """
    size_string = size_string.replace("'","").replace("(","").replace(")","").replace(" ","").split(",")
    image_size = (int(size_string[0]),int(size_string[1]))
    return image_size

def ImageReconstruction(DNA_column=None,min_value_column=None,image_size_column=None):
    """
    *Construct a color image* 
        parameters:
        ----------
            :param DNA_column:
                type: df column
                default: None.
                A column of DNA strings
            :param min_value_column:
                type: df column
                default: None.
                A column of minimum value (encoded to DNA) used to encode each DNA string
            :param image_size_column:
                type: df column
                default: None.
                A column of the dimensions of original image the DNA string came from

        returns:
        --------
        A np.ndarray color image
        """
    if type(image_size_column.iloc[0]) == str:
        image_size_list = [Str2Tuple(i) for i in image_size_column]
    elif type(image_size_column.iloc[0]) == tuple:
        image_size_list = list(image_size_column)
        
    if Normalized==True:
        DNA_strings = list(DNA_column)
        min_values_list = list(min_value_column)
        joint_list = [seq2splits(dna_str.replace(" ",''),k_mer_size).split(" ") for dna_str in DNA_strings]
        dna_list= []
        for dna,min_value,image_size in zip(joint_list,min_values_list,image_size_list):
            dna_list.append(min_value+"".join(dna))   
        image_matrix = JoinChunnles(dna_list,image_size)
        
    if Normalized==False:
        DNA_strings = list(DNA_column)
        joint_list = [seq2splits(dna_str.replace(" ",''),k_mer_size).split(" ") for dna_str in DNA_strings]
        dna_list= []
        for dna,image_size in zip(joint_list,image_size_list):
            dna_list.append("".join(dna))   
        image_matrix = JoinChunnles(dna_list,image_size)
        
    return image_matrix  