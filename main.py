#!/usr/bin/env python

#########################################################################################
#                                                                                       #
#   CODE BY: JESSY LIAO                                                                 #
#   CSCI470: FINAL PROJECT                                                              #
#                                                                                       #
#########################################################################################

import sys
# MAIN: calls all other functions 
def main(argv):

    input_file = ""

    # Read in arguments, sets inputfile and number of k iterations for simualted annealing
    if len(argv) != 5:
        print('Usage: %s [input_file]' % argv[0])
        sys.exit(0)
    else:
        source_data = argv[1]
        target_data = argv[2]

        source_lang = argv[3]
        target_lang = argv[4]


########################################################################################
        """
        READ_DATA:

        Process Data: (for both source and target data)
            - opens file
            - read file
            - strip data
            - split newlines
        """
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

    


    

    sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)
