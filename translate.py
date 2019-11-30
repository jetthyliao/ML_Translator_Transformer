import sys
import re
import math

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from train import nopeak_mask, create_fields, get_model

def init_vars(src, model, SRC, TRG, opt): 
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    if opt['device'] == 0:
        outputs = outputs.cuda()
    
    trg_mask = nopeak_mask(opt, 1)
    
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt['k'])
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt['k'], opt['max_length']).long()
    if opt['device'] == 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt['k'], e_output.size(-2),e_output.size(-1))
    if opt['device'] == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    #print('start')
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    #print('next1')
    ind = None
    for i in range(2, opt['max_length']):
        #print('i: ' + str(i))

        trg_mask = nopeak_mask(opt, i)

        out = model.out(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt['k'])
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long)
        if opt['device'] == 0:
            sentence_lengths = sentence_lengths.cuda()
        #print('sentence_lengths: ')
        #print(sentence_lengths)
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt['k']:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    #print('next2')
    if ind is None:
        #print("ind")
        length = (outputs[0]==eos_tok).nonzero()[0]
        #print(length)
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        #print("not ind")
        length = (outputs[ind]==eos_tok).nonzero()[0]
        #print(length)
        #print(outputs)
        #print([tok for tok in outputs[ind][1:length]])
        #print("before return")
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

def main(argv):
    options = {'weight_path'            : 'weights',
                'source_language'       : 'en',
                'target_language'       : 'de',
		'k'			: 3,
		'max_length' 		: 80,
                "d_model"		: 512,
                "n_layers"		: 6,
                "heads"			: 8,
                "dropout"		: 0.1,
                'device'                : 1 }

    if torch.cuda.is_available():
        options["device"] = 0

    SOURCE, TARGET = create_fields(options)
    model = get_model(options, len(SOURCE.vocab), len(TARGET.vocab))

    # test_sentence = 'provides a sample action list for use with job that are sent to the.'
    test_sentence = 'indicates if the printer is on.'
    # test_sentence = 'indicates.'
    # test_sentence = 'CMR Processing Modes'
    
    print('test_sentence: ' + test_sentence)

    model.eval()
    indexed = []
    sentence = SOURCE.preprocess(test_sentence)

    #print(sentence)

    for token in sentence:
        if SOURCE.vocab.stoi[token] != 0:
            indexed.append(SOURCE.vocab.stoi[token])
        else:
            print(token)
            print('oops')
            quit()
    sentence = Variable(torch.LongTensor([indexed]))
    if options['device'] == 0:
        sentence = sentence.cuda()

    #print(sentence)
    sentence = beam_search(sentence, model, SOURCE, TARGET, options)
    #print(sentence)

    dictionary = {' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}
    regex = re.compile("(%s)" % "|".join(map(re.escape, dictionary.keys())))
    print('translation: ' + str(regex.sub(lambda mo: dictionary[mo.string[mo.start():mo.end()]], sentence)))


if __name__ == '__main__':
        main(sys.argv)


