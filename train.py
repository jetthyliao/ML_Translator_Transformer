#########################################################################################
#																						#
#	CODE BY: JESSY LIAO / JOSEPH KIM / COURTNEY RICHARDSON / MATT CLOUGH				#
#	BASE CODE FOUND HERE: https://github.com/SamLynnEvans/Transformer					#
#	CSCI470: FINAL PROJECT																#
#																						#
#########################################################################################
import sys
import os
import time
import re
import time

import spacy
import numpy as np
import pandas as pd
import dill as pickle

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext
from torchtext import data

from transformer import Transformer

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
"""
NEED TO RUN THESE COMMANDS TO DOWNLOAD SPACY LANGUAGE LIBRARIES FOR THIS CLASS TO WORK
	python3 -m spacy download en
	python3 -m spacy download de
"""
class Tokenize(object):
	def __init__(self, lang):
		self.nlp = spacy.load(lang)
	
	def tokenizer(self, sentence):
		sentence = re.sub(
			r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
		sentence = re.sub(r"[ ]+", " ", sentence)
		sentence = re.sub(r"\!+", "!", sentence)
		sentence = re.sub(r"\,+", ",", sentence)
		sentence = re.sub(r"\?+", "?", sentence)
		sentence = sentence.lower() 
		return [token.text for token in self.nlp.tokenizer(sentence) if token.text != " "]

def create_fields(options):
	print("LOADING TOKENIZER -BEEP BOOP-")

	t_source = Tokenize(options["source_language"])
	t_target = Tokenize(options["target_language"])

	SOURCE = data.Field(lower = True, tokenize=t_source.tokenizer)
	TARGET = data.Field(lower = True, tokenize=t_target.tokenizer, init_token='<sos>', eos_token='<eos>')

	if options["weight_path"] is not None:
		try:
			print("LOADING OLD FIELDS -BING BOTT-")
			SOURCE = pickle.load(open(options["weight_path"] + "/SOURCE.pkl", 'rb'))
			TARGET = pickle.load(open(options["weight_path"] + "/TARGET.pkl", 'rb'))
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
class MyIterator(data.Iterator):
	def create_batches(self):
		if self.train:
			def pool(d, random_shuffler):
				for p in data.batch(d, self.batch_size * 100):
					p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
					for b in random_shuffler(list(p_batch)):
						yield b
			self.batches = pool(self.data(), self.random_shuffler)
		else:
			self.batches = []
			for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
				self.batches.append(sorted(b, key=self.sort_key))

global max_source_in_batch, max_target_in_batch

def batch_size_fn(new, count, sofar):
	"Keep augmenting batch and calculate total number of tokens + padding."
	global max_source_in_batch, max_target_in_batch
	if count == 1:
		max_source_in_batch = 0
		max_target_in_batch = 0
	max_source_in_batch = max(max_source_in_batch,	len(new.source))
	max_target_in_batch = max(max_target_in_batch,	len(new.target) + 2)
	source_elements = count * max_source_in_batch
	target_elements = count * max_target_in_batch
	return max(source_elements, target_elements)

def get_len(train):
	for i, b in enumerate(train):
		pass
	return i

def create_dataset(options, SOURCE, TARGET):
	raw_data = {'source': [line for line in options['source_data']], 'target': [line for line in options['target_data']]}
	dataframe = pd.DataFrame(raw_data, columns=['source', 'target'])

	mask = (dataframe['source'].str.count(' ') < options['max_string_length']) & (dataframe['target'].str.count(' ') < options['max_string_length'])
	dataframe = dataframe.loc[mask]

	dataframe.to_csv('translate_transformer_temp.csv', index = False)

	data_fields = [('source', SOURCE), ('target', TARGET)]
	train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

	if options["device"] == 0:
		device_string = 'cuda'
	else:
		device_string = 'cpu'

	train_iter = MyIterator(train, batch_size=options['batchsize'], device=device_string, repeat=False, sort_key=lambda x: (len(x.source), len(x.target)), batch_size_fn=batch_size_fn, train=True, shuffle=True)

	os.remove('translate_transformer_temp.csv')

	if options['weight_path'] is None:
		SOURCE.build_vocab(train)
		TARGET.build_vocab(train)
		if options['checkpoint'] > 0:
			try:
				os.mkdir('weights')
			except:
				print('weights folder already exists, run program with path to weights folder to load them')
				quit()
			pickle.dump(SOURCE, open('weights/SOURCE.pkl', 'wb'))
			pickle.dump(TARGET, open('weights/TARGET.pkl', 'wb'))

	options['source_pad_index'] = SOURCE.vocab.stoi['<pad>']
	options['target_pad_index'] = TARGET.vocab.stoi['<pad>']

	options['train_length'] = get_len(train_iter)

	return train_iter


#######################################################################################
"""
GET MODEL: 
get_model(options, len(source vocab), len(target vocab))
creates a Transformer class object
also loads any weights that were created in prior training runs
"""
def get_model(options, source_vocab, target_vocab):
	model = Transformer(source_vocab, target_vocab, options["d_model"], options["n_layers"], options["heads"], options["dropout"])

	if options["weight_path"] != None: 
		print("LOADING PRETRAINED WEIGHT -DING DONG-")
		model.load_state_dict(torch.load(options["weight_path"]+"/model_weights"))
	else:
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	if options["device"] == 0:
		model = model.cuda()

	return model


#######################################################################################
"""
TRAIN_MODEL:
train_model(options, model)
"""
def nopeak_mask(options, size):
	np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
	np_mask =  Variable(torch.from_numpy(np_mask) == 0)
	if options['device'] == 0:
	  np_mask = np_mask.cuda()
	return np_mask

def create_masks(options, source, target): 
	source_mask = (source != options['source_pad_index']).unsqueeze(-2)

	if target is not None:
		target_mask = (target != options['target_pad_index']).unsqueeze(-2)
		size = target.size(1) # get seq_len for matrix
		np_mask = nopeak_mask(options, size)
		if target.is_cuda:
			np_mask.cuda()
		target_mask = target_mask & np_mask
	else:
		target_mask = None
	return source_mask, target_mask

def train_model(options, model):
	model.train()
	start = time.time()
	if options['checkpoint'] > 0:
		cptime = time.time()
				 
	for epoch in range(options['epochs']):
		total_loss = 0
		print("	  %dm: epoch %d [%s]  %d%%	loss = %s" % ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
		
		if options['checkpoint'] > 0:
			torch.save(model.state_dict(), 'weights/model_weights')

		for i, batch in enumerate(options['train']): 
			source = batch.source.transpose(0,1)
			target = batch.target.transpose(0,1)
			target_input = target[:, :-1]
			source_mask, target_mask = create_masks(options, source, target_input)
			predictions = model(source, target_input, source_mask, target_mask)
			ys = target[:, 1:].contiguous().view(-1)
			options['optimizer'].zero_grad()
			loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), ys, ignore_index=options['target_pad_index'])
			loss.backward()
			options['optimizer'].step()
			if options['SGDR'] == True: 
				options['sched'].step()
			
			total_loss += loss.item()
			if (i + 1) % options['printevery'] == 0:
				p = int(100 * (i + 1) / options['train_length'])
				avg_loss = total_loss/options['printevery']
				print("	  %dm: epoch %d [%s%s]	%d%%  loss = %.3f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
				total_loss = 0
			
			if options['checkpoint'] > 0 and ((time.time()-cptime)//60) // options['checkpoint'] >= 1:
				torch.save(model.state_dict(), 'weights/model_weights')
				cptime = time.time()
   
		print("%dm: epoch %d [%s%s]	 %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))
		
		
#######################################################################################
# MAIN: calls all other functions 
def main(argv):
	input_file = ""

	# Read in arguments, sets inputfile
	if len(argv) == 2:
		data_filename = argv[1]
		weight_path = None
	elif len(argv) == 3:
		data_filename = argv[1]
		weight_path = argv[2]
	else:
		print('Usage: %s [input_file]' % argv[0])
		sys.exit(0)

	options = {"data_filename"		: data_filename,
				"weight_path"		: weight_path,
				"source_data"		: {},
				"target_data"		: {},
				"source_language"	: "en",
				"target_language"	: "de",
				"epochs"			: 1,
				"d_model"			: 512,
				"n_layers"			: 6,
				"heads"				: 8,
				"dropout"			: 0.1,
				"batchsize"			: 1500,
				"printevery"		: 2,
				"lr"				: 0.0001,
				"max_string_length"	: 80,
				"checkpoint"		: 5,
				"device"			: 1,
				"SGDR"				: 'store_true'}

	if torch.cuda.is_available():
		options["device"] = 0

	options["source_data"], options["target_data"] = read_file(options)

	SOURCE, TARGET = create_fields(options)

	options["train"] = create_dataset(options, SOURCE, TARGET)

	model = get_model(options, len(SOURCE.vocab), len(TARGET.vocab))

	options["optimizer"] = torch.optim.Adam(model.parameters(), lr=options["lr"], betas=(0.9,0.98), eps=1e-9)

	if options["SGDR"] == True: 
		print("something")

	train_model(options, model)


if __name__ == '__main__':
	main(sys.argv)
