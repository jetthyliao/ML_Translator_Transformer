CSCI470 Project
Product Domain Machine Translation with Ricoh USA

Jessy Liao - jessyliao@mymail.mines.edu
Courtney Richardson - courtneyrichardson@mymail.mines.edu
Matt Clough - mclough@mymail.mines.edu
Joseph Kim - josephkim@mymail.mines.edu

Proof of concept python script using Transformer model from 'Attention is all you need' by Vaswani et al.
https://arxiv.org/abs/1706.03762

Base code from Sam Lynn Evans who implemented the Transformer model. Took his implementation and modified for our purpose.
https://github.com/SamLynnEvans/Transformer

Usage:
	Run train.py with input_data file; if weights/ folder exists from previous training, pass in path to weights/ as second argument.
	Can change epochs and other hyperparameters in main of train.py; options dictionary.
	Training data (SOURCE.pkl, TARGET.pkl, model_weights) get saved to weights/ folder (made in same folder as script).
	
	Run translate.py with weights/ folder in same directory.
	First argument is text file with newline seperated sentences.
	Prints source and target sentences. Writes translated sentences to new file in the same folder as script.
