#!/usr/bin/env python

#########################################################################################
#   	    										#
#   CODE BY: JESSY LIAO / JOSEPH KIM / COURTNEY RICHARDSON / MATT CLOUGH                #
#   CSCI470: FINAL PROJECT				    				#
#   											#
#########################################################################################

import sys
import os

import spacy
import re
import time
import torch
#from Models import get_model
#from Process import *
import torch.nn.functional as F
#from Optim import CosineWithRestarts
#from Batch import create_masks
import dill as pickle

import pandas as pd
import torchtext
from torchtext import data
#from Batch import MyIterator, batch_size_fn


########################################################################################
"""
READ_DATA:

Process Data: (for both source and target data)
	- opens file
	- read file
	- strip data
	- split newlines
"""
def read_file(options):
    try:		
        file_lines = open(options["data_filename"], 'r').read().strip().split('\n')

        source_column = []
        target_column = []
        
        for line in file_lines:
            split_line = line.split('\t')
            print(split_line[0])
            print(split_line[1])
            print()
            source_column.append(split_line[0])
            target_column.append(split_line[1])
        return (source_column, target_column)
    except:
        print("error: '" + options["data_filename"] + "' file not found")
        quit()
	
#######################################################################################
"""
CREATE FIELDS: 

Create tokenize object and initilize with languages
	- self.nlp = spacy.load(lang) for both en and fr

TorchText.data.field(lower,tokenize, init_token, eos_token)
	- lower: lowercase text 
	- tokenize: function used to tokenize string (use spacy or use custom)
	- init_token: token prepended (only for target data)
	- eos_token: token append at end (only for target data)

If existing data, load existing data (Uses pickle). This overwrites the above code. 
	- load_weights is the path to the pkl file 
	- this is old weight data to use if we have weights available
"""
class Tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
    
    def tokenizer(spacy_load, sentence):
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower() 
        return [token.text for token in space_load.tokenizer(sentence) if token.text != " "]

def create_fields(options):

    print("LOADING TOKENIZER -BEEP BOOP-")

    t_source = Tokenize(options["source_language"])
    t_target = Tokenize(options["target_language"])

    TARGET = data.Field(lower = True, tokenize=t_target.tokenizer, init_token='<sos>', eos_token='<eos>')
    SOURCE = data.Field(lower = True, tokenize=t_source.tokenizer)

    if options["load_weights"] is not None:
        try:
            print("LOADING OLD FIELDS -BING BOTT-")
            TARGET = pickle.load(open(f'{option["load_weights"]}/TARGET.pkl', 'rb'))
            SOURCE = pickle.load(open(f'{option["load_weights"]}/SOURCE.pkl', 'rb'))
        except:
            print("error opening SOURCE and TARGET pickles, please ferment cucumber longer")
    
    return (SOURCE, TARGET)
 

#######################################################################################
"""
CREATE DATASET: create datseta and iterator
	
Create dictionary
	- keys = source, target
	- value = list from read source data and read target data

Create dataframe (format data to a csv)
	- remove long sentences in dataframe

TorchText.data.TabularDataset(csv, format, fields)
	- CSV FILE
	- format is just what file type
	- fields = list of tuples(str, Field)
		- Field should be same order as columns in CSV

	- Uses field tokenizer on the csv file data to create a tabular dataset (me thinks that is what happening)

Create Iterator with MyIterator() (copied from github)
	- batch_size: measured as number of tokens fed in each iteration (number of sentences)
	- device: cpu or gpu
	- sort_key: for any x sort_key = tuple(len x.src, len x.trg)
	- batch_size_fn: adds a new sentence to the batch and returns the new batch size (number of sentences)
	- train: represnts train set or not
	- shuffle: whether to shuffle example between epochs

	BATCH = list of words, words are represented as the embedding vectors + positional encoding

If there are no preexisting load data (Step in create field)
	- it will build vocabulary on both source and target
	- build_vocab takes a word (token) and gives it a numerical value
	- use stoi and itos to see the word/number

If you want checkpoints (save the weights) make a file and pickle dump the built vocabulary into the file

Get the index for the pad value for later use

Get the length of the iterator and then returns the iterator itself 
"""
#######################################################################################
"""
GET MODEL: 

get_model(opt, len(source vocab), len(target vocab))
	- d_model: dimension of embedding vector and layers (Nx)
	- 
"""

# MAIN: calls all other functions 
def main(argv):

    input_file = ""

    # Read in arguments, sets inputfile
    if len(argv) == 2:
        data_filename = argv[1]
        weight_path = None
    elif len(argv) == 3:
        data_filename = argv[1]
        weights_path = argv[2]
    else:
        print('Usage: %s [input_file]' % argv[0])
        sys.exit(0)
 
    
    options = {"data_filename"		: data_filename,
               "source_data"            : {},
               "target_data"            : {},
               "source_language"	: "en",
               "target_language"	: "de",
               "epochs"			: 2,
               "d_model"		: 512,
               "n_layers"		: 6,
               "heads"			: 8,
               "dropout"		: 0.1,
               "batchsize"		: 1500,
               "printevery"		: 100,
               "lr"			: 0.0001,
               "max_strlen"		: 80,
               "checkpoint"		: 0,
               "device"			: 1,
               "load_weights"           : weight_path}

    if torch.cuda.is_available():
        options["device"] = 0

    options["source_data"], options["target_data"] = read_file(options)

    SOURCE, TARGET = create_fields(options)   

    sys.exit(1)



if __name__ == '__main__':
    main(sys.argv)
