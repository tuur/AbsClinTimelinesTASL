from allennlp.modules.elmo import Elmo, batch_to_ids
from glove import Corpus, Glove
import sklearn.cluster
import pickle
from collections import Counter
from functools import reduce
import torch, sys, random, os, glob, re
import numpy as np
import torch.autograd as autograd
import datetime, time, random
from baseconvert import base
import lib.losses as losses
from lib.agreement import Evaluation, make_timeline
from lib.i2b2 import  write_annotation_to_event_in_doc, clear_i2b2_file_from_btime_annotations
from lib.EventTimeAnnotation import CalenderDuration
from lib.utils import Stats
from torch import nn
random.seed(0)



from lib.data import tokenize
from copy import deepcopy, copy
from shutil import copyfile
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import dateutil

import matplotlib.pyplot as plt
from lib.EventTimeAnnotation import TimeLineTask


class SequenceEncoder(nn.Module):

    def __init__(self, cfg, docs):
        super(SequenceEncoder, self).__init__()
        self.cfg, self.unk_token, self.word_dim,self.out_dim, self.elmo, self.wembedding, self.lstm, self.cnn = cfg, '__unk__', 0, 0, 0, 0, 0, 0
        if cfg['elmo_options'] and cfg['elmo_weights']:
            self.elmo = Elmo(cfg['elmo_options'], cfg['elmo_weights'], 2, dropout=cfg['dropout']).float()
            self.word_dim += self.elmo.get_output_dim() #min(, cfg['elmo_dims_used'])
        if cfg['wemb_dim'] or cfg['glove']:
        #    print('!')
            if cfg['glove']:
                glove_model = Glove.load(cfg['glove'])
                num_words, emb_dim = len(glove_model.dictionary), len(glove_model.word_vectors[0])
                emb_matrix = torch.Tensor(glove_model.word_vectors)
                self.windex  = glove_model.dictionary
                self.windex[self.unk_token] = len(self.windex)
                mean_vector = torch.mean(emb_matrix, dim=0)
                emb_matrix = torch.cat([emb_matrix, mean_vector.view(1, emb_dim)])
                self.wembedding = torch.nn.Embedding(len(self.windex), emb_dim)
                self.wembedding.weight = nn.Parameter(emb_matrix)
                self.word_dim += emb_dim
            else:
                self.windex, self.cindex = self.setup_vocabularies([doc.tokens for doc in docs])
                self.wembedding = torch.nn.Embedding(len(self.windex), cfg['wemb_dim'])
                self.word_dim += cfg['wemb_dim']
        if cfg['lstm_dim']:
            self.lstm = nn.LSTM(self.word_dim, self.cfg['lstm_dim'], bidirectional=False)
            self.out_dim += self.cfg['lstm_dim']
        if cfg['cnn_window_sizes'] and cfg['cnn_filters']:
            self.relu = nn.LeakyReLU()
            self.window_sizes, self.num_cnn_filters = [int(i) for i in cfg['cnn_window_sizes'].split(',')], cfg['cnn_filters']
            self.cnn = nn.ModuleList([
                nn.Conv2d(1, self.num_cnn_filters, [window_size, self.word_dim], padding=(window_size - 1, 0))
                for window_size in self.window_sizes
            ])
            self.out_dim += self.num_cnn_filters * len(self.window_sizes)

        self.pad_embedding = torch.zeros(self.word_dim,requires_grad=True).view(1,self.word_dim).double()

    def embed_doc(self, doc):
        tokens = [t if not t == '\n' else '</S>' for t in doc.tokens]
        embs = False
        if self.elmo:
            #with torch.no_grad():
                character_ids = batch_to_ids([tokens]).cuda(self.cfg['gpu'])
                embs = self.elmo(character_ids)['elmo_representations'][1][0].double() #.detach() #[:self.cfg['elmo_dims_used']]
                del character_ids
        if self.wembedding:
            indices =[self.windex[w] if w in self.windex else self.windex[self.unk_token] for w in tokens]
            word_ids = torch.LongTensor(indices).cuda(self.cfg['gpu'])
            wembs = self.wembedding(word_ids)
            embs = wembs if not isinstance(embs, torch.Tensor) else torch.cat((embs, wembs),dim=1)
            del word_ids
        return embs

    def forward(self, list_of_list_of_token_ids, embedded_doc):
        max_seq_length = max([len(l) for l in list_of_list_of_token_ids])
        emb_list = []
        for tok_list in list_of_list_of_token_ids:
            if len(tok_list) < max_seq_length:
                pad_emb = self.pad_embedding.repeat(max_seq_length - len(tok_list),1).cuda(self.cfg['gpu'])
                emb_slice =  torch.index_select(embedded_doc,0 ,torch.LongTensor(tok_list).cuda(self.cfg['gpu'])) #embedded_doc[tok_list[0]:tok_list[-1]]
                emb_seq = torch.cat((pad_emb,emb_slice))
            else:
                emb_seq = torch.index_select(embedded_doc,0 ,torch.LongTensor(tok_list).cuda(self.cfg['gpu']))
            emb_list.append(emb_seq)
        embs = torch.stack(emb_list)
        enc = False
        if self.lstm:
            enc = self.lstm(embs)[0][:,-1]
        if self.cnn:
            cnn_in = torch.unsqueeze(embs, 1)
            cnn_encs = []
            for cnn_i in self.cnn:
                tmp = self.relu(cnn_i(cnn_in))
                tmp = torch.squeeze(tmp, -1)
                tmp = torch.nn.functional.max_pool1d(tmp, tmp.size(2))
                cnn_encs.append(tmp)
            cnn_enc = torch.cat(cnn_encs, 2)
            cnn_enc = cnn_enc.view(cnn_enc.size(0), -1)
            enc = cnn_enc if not isinstance(enc, torch.Tensor) else torch.cat((enc, cnn_enc), dim=1)
        del emb_list
        del embs
        return enc


    def setup_vocabularies(self, texts):
        self.word_frequencies = Counter([token for text in texts for token in text])
        self.char_frequencies = Counter([c for text in texts for c in text])
        new_word_frequencies = {w:freq for w,freq in self.word_frequencies.items() if freq > self.cfg['unk_threshold']}
        new_char_frequencies = {c:freq for c,freq in self.char_frequencies.items() if freq > self.cfg['unk_threshold']}
        cindex = {w: i for i, w in enumerate(list(new_char_frequencies.keys()) + [self.unk_token])}
        windex = {w: i for i, w in enumerate(list(new_word_frequencies.keys()) + [self.unk_token])}
        return windex, cindex



class TextEncoder(nn.Module):

    def __init__(self, cfg, docs):
        super(TextEncoder, self).__init__()
        self.unk_token, self.span_dim, self.cfg  = '__unk__', 0, cfg
        self.windex, self.cindex = self.setup_vocabularies([doc.tokens for doc in docs])
        self.elmo = Elmo(cfg['elmo_options'], cfg['elmo_weights'], 2, dropout=0) if cfg['elmo_options'] and cfg['elmo_weights'] else False
        self.span_dim = self.span_dim + self.elmo.get_output_dim() if self.elmo else self.span_dim
        self.wembedding = torch.nn.Embedding(len(self.windex), cfg['wemb_dim']) if cfg['wemb_dim'] else False
        self.span_dim = self.span_dim + cfg['wemb_dim'] if (self.wembedding and not cfg['lstm_dim']) else self.span_dim

        self.lstm = nn.LSTM(self.wembedding.embedding_dim, self.cfg['lstm_dim'], bidirectional=True) if self.cfg['lstm_dim'] and self.wembedding else False
        self.span_dim = self.span_dim + 2 * self.cfg['lstm_dim'] if self.lstm else self.span_dim
        self.linear_reshaper = nn.Linear(self.span_dim, cfg['span_dim']) if cfg['linear_reshaper'] else False
        self.dropout = nn.Dropout(cfg['dropout'])
        self.tied_elmo = 'elmo' in cfg['tied_parameters_regex']

    def preproc_text(self,text):
        return [w if w in self.windex else self.unk_token for w in text]

    def forward(self, doc):
        """ Inputs a text (list of strings) """
        text = doc.tokens
        return self.text_forward(text, doc)

    def text_forward(self, list_of_strings, doc=None):
        encoded = {}
        if self.elmo:
            if doc.has_elmo_encoding and self.tied_elmo:
                encoded['elmo'] = doc.elmo_encoding
            else:
                with torch.no_grad():
                    character_ids = batch_to_ids([list_of_strings]).cuda(self.cfg['gpu'])
                    encoded['elmo'] = self.elmo(character_ids)['elmo_representations'][1][0]
                if self.tied_elmo:
                    doc.elmo_encoding = encoded['elmo']
                    doc.has_elmo_encoding = True

        if self.wembedding:
            text = self.preproc_text(list_of_strings)
            token_ids = torch.LongTensor([self.windex[w] if w in self.windex else self.windex[self.unk_token] for w in text]).cuda(self.cfg['gpu'])
            wembs = self.wembedding(token_ids)
            encoded['wembs'] = wembs
            if self.lstm:
                out, _ = self.lstm(wembs.view(len(text), 1, self.cfg['wemb_dim']))
                encoded['lstm'] = out.view(len(text), self.cfg['lstm_dim']*2)
        return encoded



    def get_span_vector(self, span, doc, encoded): 

        token_ids = list(doc.span_to_tokens(span, token_index=True))
        span_vector = torch.Tensor([]).cuda(self.cfg['gpu']) #np.mean(token_ids) / len(doc.tokens)])
        if self.elmo:
            if span in doc.elmo_span_repr and self.tied_elmo:
                elmo_vec = doc.elmo_span_repr[span]
            else:
                elmo_vec = torch.mean(encoded['elmo'][token_ids[0]:token_ids[-1]+1], dim=0) # mean of tokens in the span
                if self.tied_elmo:
                    doc.elmo_span_repr[span] = elmo_vec
            span_vector = torch.cat((span_vector, elmo_vec))
        if self.wembedding:
            if self.lstm:
                lstm_v = torch.mean(encoded['lstm'][token_ids[0]:token_ids[-1]+1], dim=0)
                span_vector = torch.cat((span_vector, lstm_v ))
            else:
                wemb = torch.mean(encoded['wembs'][token_ids[0]:token_ids[-1]+1],dim=0)
                span_vector = torch.cat((span_vector, wemb))
        if self.dropout:
            span_vector = self.dropout(span_vector)
        return self.linear_reshaper(span_vector) if self.linear_reshaper else span_vector

    def setup_vocabularies(self, texts):
        self.word_frequencies = Counter([token for text in texts for token in text])
        self.char_frequencies = Counter([c for text in texts for c in text])
        new_word_frequencies = {w:freq for w,freq in self.word_frequencies.items() if freq > self.cfg['unk_threshold']}
        new_char_frequencies = {c:freq for c,freq in self.char_frequencies.items() if freq > self.cfg['unk_threshold']}
        cindex = {w: i for i, w in enumerate(list(new_char_frequencies.keys()) + [self.unk_token])}
        windex = {w: i for i, w in enumerate(list(new_word_frequencies.keys()) + [self.unk_token])}
        return windex, cindex

class ClusterRegression(nn.Module):
    def __init__(self, input_dim, min=-100, max=100, dropout=0.0, data=[],cfg={}):
        super(ClusterRegression, self).__init__()
        self.timeline_task = TimeLineTask(cfg)
        self.num_clusters = 10
        event_durations = [[self.timeline_task.calendar_duration_to_tlv(e_ann.resolve_relative_duration(e_ann.d, e_ann.s))] for d in data for e_ann in d.event_span_to_event_time_annotations.values()]
        self.kmeans = sklearn.cluster.KMeans(self.num_clusters).fit(event_durations)
        self.clusters = nn.Parameter(torch.FloatTensor([0.0] + [v[0] for v in self.kmeans.cluster_centers_] + [max]), requires_grad=False)
        print('regression clusters', self.clusters)
        self.in_to_h = nn.Linear(input_dim,input_dim // 2)
        self.h_to_out = nn.Linear(input_dim // 2, self.num_clusters + 2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        h = self.dropout(self.relu(self.in_to_h(input)))
        cluster_scores = self.relu(self.h_to_out(h))
        cluster_probabilities = self.softmax(cluster_scores)

        if len(cluster_probabilities.size()) == 1:
            return cluster_probabilities.dot(self.clusters)
        else:
            return torch.mv(cluster_probabilities, self.clusters)



class LinearRegressor(nn.Module):
    """ A module that takes a tensor of shape N and outputs a single output value (with a minimum and maximum value)"""

    def __init__(self, input_dim, min=-100, max=100, dropout=0.0, data=[], cfg={}):
        super(LinearRegressor, self).__init__()
        self.input_dim, self.min, self.max =  input_dim, min, max
        self.in_to_h = nn.Linear(input_dim,input_dim // 2)
        self.h_to_out = nn.Linear(input_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, input):
        h = self.dropout(self.relu(self.in_to_h(input)))
        out = self.h_to_out(h) #* (self.max - self.min) + self.min
        return out

class X3Regressor(nn.Module):
    """ A module that takes a tensor of shape N and outputs a single output value (with a minimum and maximum value)"""

    def __init__(self, input_dim, min=-100, max=100, dropout=0.0, data=[], cfg={}):
        super(X3Regressor, self).__init__()
        self.input_dim, self.min, self.max =  input_dim, min, max
        self.in_to_h = nn.Linear(input_dim,input_dim // 2)
        self.h_to_out = nn.Linear(input_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()

    def forward(self, input):
        h = self.dropout(self.relu(self.in_to_h(self.relu(input))))
        out = self.h_to_out(h)**3 #* (self.max - self.min) + self.min

        return (2*out)**3

class SigmoidRegressor(nn.Module):
    """ A module that takes a tensor of shape N and outputs a single output value (with a minimum and maximum value)"""

    def __init__(self, input_dim, min=-100, max=100, dropout=0.0, data=[], cfg={}):
        super(SigmoidRegressor, self).__init__()
        self.input_dim, self.min, self.max =  input_dim, min, max
        self.in_to_h = nn.Linear(input_dim,input_dim // 2)
        self.h_to_out = nn.Linear(input_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # extra layer decreases training time (since more parameters)
        # intermediate relu activations make it difficult to get out of local optima (even leaky relu) / idem for sigmoid except that relu converges faster
        h = self.dropout(self.relu(self.in_to_h(input))) # RELU activation is much faster, but results in stuck in local minimum (all set to 0?)
        out = self.sigmoid(self.h_to_out(h)[0]) * (self.max-self.min) + self.min
        return out

class PowerRegression(nn.Module):

    def __init__(self,input_dim, min=0, max=1.0, n=25, dropout=0.0, data=[], cfg={}):
        super(PowerRegression, self).__init__()
        self.input_dim, self.min, self.max, self.n = input_dim, min, max, n
        self.classes = nn.Parameter(torch.FloatTensor([0.0] + [np.exp(i) for i in range(self.n)]), requires_grad=False)
        self.in_to_h = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.h_to_out = nn.Linear(input_dim, self.n + 1,bias=False) # // 2, self.num_digits)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def normalizer(self,x):
        return (x*(self.max-self.min)) / torch.max(self.classes) + self.min

    def forward(self, input):
        h = self.dropout(self.relu(input))
        class_scores = self.relu(self.h_to_out(h))
        raw_output = self.classes.dot(class_scores)
        print(raw_output)
        normalized_output = self.normalizer(raw_output)
        return normalized_output

class SoftmaxPowerRegression(nn.Module):

    def __init__(self,input_dim, min=0, max=1.0, n=10, dropout=0.0, data=[], cfg={}):
        super(SoftmaxPowerRegression, self).__init__()
        self.input_dim, self.min, self.max, self.n = input_dim, min, max, n
        self.classes = nn.Parameter(torch.FloatTensor([0.0] + [np.exp(i) for i in range(self.n)] + [max]), requires_grad=False)
        self.in_to_h = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.h_to_out = nn.Linear(input_dim, self.n + 2,bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def normalizer(self,x):
        return (x*(self.max-self.min)) / torch.max(self.classes) + self.min

    def forward(self, input):
        h = self.dropout(self.relu(input))
        class_scores = self.relu(self.h_to_out(h))
        class_probabilities = self.softmax(class_scores)
        if len(class_probabilities.size()) == 1:
            raw_output = class_probabilities.dot(self.classes)
        else:
            raw_output = torch.mv(class_probabilities, self.classes)
        normalized_output = self.normalizer(raw_output)
        return normalized_output.view(normalized_output.size()[0],1)



class NaryRegressor(nn.Module):
    """A module that takes a tensor of shape N, and outputs a single output value (with a possible minimum and maximum value) by going through an n-ary encoding layer"""

    def __init__(self, input_dim, min=0, max=1000000, n=10, dropout=0.0, data=[], cfg={}):
        super(NaryRegressor, self).__init__()
        self.input_dim, self.min, self.max, self.n = input_dim, min, max, n
        self.digit_range = len(base(self.max, 10, self.n))

        pos_conversion, neg_conversion = torch.FloatTensor([self.n**digit for digit in range(self.digit_range, 1, -1)]), torch.FloatTensor([-self.n**digit for digit in range(self.digit_range, 1, -1)])
        conversion_vector = torch.FloatTensor([1.0])
        if min < 0:
            conversion_vector = torch.cat((conversion_vector, neg_conversion))
        if max > 0:
            conversion_vector = torch.cat((conversion_vector, pos_conversion))
        self.num_digits = len(conversion_vector)
        self.conversion_vector = nn.Parameter(conversion_vector, requires_grad=False)
        self.in_to_h = nn.Linear(input_dim, input_dim // 2)
        self.h_to_out = nn.Linear(input_dim, self.num_digits)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        h = self.dropout(self.relu(input))
        digit_scores = self.h_to_out(h)
        digit_values = self.softmax(digit_scores)
        interpreted_digits = self.conversion_vector*digit_values
        value = sum(interpreted_digits)
        return value


def read_cfg(cfg, name):
    """Read task specific config attributes and set them as the normal cfg value withing (within each task model)"""
    new_cfg = {}
    for key in cfg:
        if key[:len(name)] == name:
            new_cfg[key[len(name)+1:]] = cfg[key]
        else:
            new_cfg[key] = cfg[key]
    return new_cfg

class SpanRegressionTaskModel(nn.Module):
    """ A model that predicts a value in a range(min, max) for text spans"""

    def __init__(self, name, min, max, cfg, data):
        super(SpanRegressionTaskModel, self).__init__()
        self.name, self.min, self.max, self.cfg = name, min, max, read_cfg(cfg,name)
        self.text_encoder = TextEncoder(self.cfg, [doc for doc in data])#.cuda(self.cfg['gpu'])
        self.regressor = SigmoidRegressor(input_dim=self.text_encoder.span_dim, min=self.min,max=self.max) if self.cfg['regressor_type']=='SigmoidRegressor' \
                                                else NaryRegressor(input_dim=self.text_encoder.span_dim, min=self.min,max=self.max,n=self.cfg['number_arity'])
        self.cuda(self.cfg['gpu'])

    def pred_spans_in_doc(self, spans, doc):
        encoded = self.text_encoder(doc)
        preds = []
        for span in spans:
            vec = self.text_encoder.get_span_vector(span, doc, encoded)
            value = self.regressor(vec)
            preds.append(value)
        return preds


class AbsoluteTimelineModel(nn.Module):

    def __init__(self, cfg, data):
        super(AbsoluteTimelineModel, self).__init__()
        self.cfg = cfg
        self.model_dir= "./out/" + datetime.datetime.now().strftime("%Y-%m-%d") + "/" + cfg["experiment_name"]
        self.timeline_task = TimeLineTask(cfg)
        self.build_model(cfg, data)

    def get_number_of_parameters(self, cfg, trained=True):
        num = 0
        parameter_iterator = self.get_parameters() if not trained else self.trained_parameters(cfg)
        for params in parameter_iterator:
                num+=reduce((lambda x, y: x * y), params.shape)
        return num

    def trained_parameters(self, cfg):

        params = []
        including_regex = cfg['trained_parameters_regex']
        excluding_regex = cfg['tied_parameters_regex']
        print('IN:',including_regex)
        print('OUT:',excluding_regex)
        for name,param in self.named_parameters():
            add=False
            if len(re.findall(including_regex, name)) > 0 and len(re.findall(excluding_regex,name)) == 0:
                params.append(param)
                add=True
            if cfg['verbose']:
                print('train param', name, add)
        return params

    def save_model(self, path=None):
        if not path:
            path=self.model_dir + '/model.p'
        print ('saving model', path)
        init_time = time.time()
        with open(path, 'wb') as f:
           pickle.dump(self, f)
        print('saved t:', round(time.time() - init_time, 2), 's')

    def train_model(self, cfg, data):
        self.train_model_dir = self.model_dir + '/' + cfg['train_prefix'] + '/'
        self.cfg = cfg
        self.to(cfg['gpu'])
        optimizer = torch.optim.Adam(self.trained_parameters(cfg), lr=cfg['lr']) #, amsgrad=True)
        losses = set([w.replace(' ','') for w in cfg['losses'].split(',')])
        metrics = cfg['train_metrics'].replace(' ','').split(',')

        dev_data, train_data = data[:cfg['development_set_size']], data[cfg['development_set_size']:]
        print('dev', sorted([d.id for d in dev_data]))

        print('starting train:', len(train_data), 'dev:',len(dev_data))
        print('losses',cfg['losses'])
        print('lr',cfg['lr'])
        print('parameters trained:',cfg['trained_parameters_regex'])
        print('Number of parameters trained:',self.get_number_of_parameters(cfg, trained=True))

        if cfg['output_train_pred_xmls']:
            train_pred_file_dir = self.train_model_dir + '/intermediate_preds/train/'
            dev_pred_file_dir = self.train_model_dir + '/intermediate_preds/dev/'
            os.makedirs(train_pred_file_dir + '/xml/')
            os.makedirs(train_pred_file_dir + '/viz/')
            os.makedirs(dev_pred_file_dir + '/xml/')
            os.makedirs(dev_pred_file_dir + '/viz/')
            out_train_doc_ids = set([doc.id for doc in train_data[:cfg['output_train_pred_xmls']]])
            out_dev_doc_ids = set([doc.id for doc in dev_data[:cfg['output_train_pred_xmls']]])
        if cfg['checkpoint']:
            checkpoint_dir = self.train_model_dir +'/checkpoints/'
            os.makedirs(checkpoint_dir)

        stats = Stats()
        early_stopping = {'lowest_dev_loss':float("inf"), 'best_model':None, 'patience_left':cfg['patience'],'best_epoch':None}
        for epoch in range(cfg['max_number_of_epochs']):
            #torch.cuda.empty_cache()
            t0 = time.time()

            if cfg['checkpoint'] and epoch % cfg['checkpoint'] == 0:
                self.save_model(checkpoint_dir + 'checkpoint_'+str(epoch)+'.p')

            # shuffle the training data at each epoch
            random.shuffle(train_data)

            # set network to train mode (needed for dropout)
            self.train()

            # for each document in the training data
            for did,doc in enumerate(train_data):

                # +++++++++++++++ TRAIN ++++++++++

                pred_file_path = train_pred_file_dir + '/xml/e'+ str(epoch) + '-' + doc.id if doc.id in out_train_doc_ids else False
                write_timeline_path = train_pred_file_dir + '/viz/e' + str(epoch) + '-' + doc.id[:-4] + '-diff.html' if doc.id in out_train_doc_ids else False

                # make predictions for the events in the document
                predicted_doc = self.pred_doc(doc, max_number_of_events=self.cfg['max_events_in_batch'])

                # calculating the loss
                loss_calculation = self.get_loss(predicted_doc['preds'], predicted_doc['gt_anns'], loss_functions=losses)

                # backpropagation
                self.zero_grad()
                #print(loss_calculation)
                loss_calculation['L'].backward()

                numpy_loss_calculation = {k:v.cpu().detach().numpy() for k,v in loss_calculation.items()}

                # clip gradients to prevent exploding gradients
                if cfg['clip']:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), cfg['clip'])
                    #torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), cfg['clip'])

                optimizer.step()

                # add the loss information to the statistics (for plotting)
                stats.add_dict(epoch_num=epoch,stat_dict=numpy_loss_calculation, prefix='train_')

                # calculate some more evaluation metrics and add them to the plot
                doc_eval = Evaluation(predicted_doc['gt_anns'], predicted_doc['pred_anns'],metrics=metrics,course=cfg['course_eval'])
                stats.add_dict(epoch_num=epoch,stat_dict=doc_eval.results,prefix='train_')

                # write the predictions to file
                if pred_file_path:
                    copyfile(doc.file_path, pred_file_path)
                    for pred_ann in predicted_doc['pred_anns']:
                        write_annotation_to_event_in_doc(pred_file_path, pred_ann)

                # write the timelines to files (in comparison to the ground truth timelines)
                if write_timeline_path:
                    event_strings = [doc.text[span[0]:span[1]] for span in predicted_doc['event_spans']]
                    make_timeline(predicted_doc['gt_anns'], predicted_doc['pred_anns'], event_strings, event_strings, write_timeline_path, max=True)
                 del predicted_doc
                del loss_calculation
                del doc_eval


            # +++++++++++++++++ DEV ++++++++++

            # set the network to evaluation mode
            self.eval()

            # for each document in the development data
            for did,doc in enumerate(dev_data):
                torch.cuda.empty_cache()

                pred_file_path = dev_pred_file_dir + '/xml/e' + str(epoch) + '-' + doc.id
                write_timeline_path = dev_pred_file_dir + '/viz/e' + str(epoch) + '-' + doc.id[:-4] + '-diff.html' if doc.id in out_dev_doc_ids else False

                # make predictions for each event in the document
                predicted_doc = self.pred_doc(doc, max_number_of_events=self.cfg['max_events_in_batch'])

                # calculate the development loss (for early stopping and checking of overfitting)
                loss_calculation = self.get_loss(predicted_doc['preds'], predicted_doc['gt_anns'], loss_functions=losses)
                numpy_loss_calculation = {k:v.cpu().detach().numpy() for k,v in loss_calculation.items()}

                # calculate evaluation statistics and add them to the plot
                doc_eval = Evaluation(predicted_doc['gt_anns'], predicted_doc['pred_anns'], metrics=metrics,course=cfg['course_eval'])
                stats.add_dict(epoch_num=epoch, stat_dict=doc_eval.results, prefix='dev_')
                stats.add_dict(epoch_num=epoch, stat_dict=numpy_loss_calculation, prefix='dev_')

                # write the predictions to xml
                if pred_file_path:
                    copyfile(doc.file_path, pred_file_path)
                    for pred_ann in predicted_doc['pred_anns']:
                        write_annotation_to_event_in_doc(pred_file_path, pred_ann)

                # write the timelines, in comparison to the ground truth timelines to file
                if write_timeline_path:
                    event_strings = [doc.text[span[0]:span[1]] for span in predicted_doc['event_spans']]
                    make_timeline(predicted_doc['gt_anns'], predicted_doc['pred_anns'], event_strings, event_strings, write_timeline_path, max=True)

                del predicted_doc
                del loss_calculation
                del doc_eval

            # add the epoch time to the statistics and plot and print them
            stats.add_dict(epoch,{'t':time.time()-t0})
            stats.pplot(self.train_model_dir +'/train_stats.html')
            stats.pprint_epoch_summary(epoch,[('train_L',6),('t',2)])

            # check if the model is overfitting to possibly stop the training (early stopping)
            if cfg['patience'] and len(dev_data) > 0:
                epoch_dev_loss = stats.epoch_stats[epoch].get_mean_value('dev_L')
                if epoch_dev_loss <= early_stopping['lowest_dev_loss']: # if loss goes down still: reset early stopping
                    early_stopping['lowest_dev_loss'],early_stopping['patience_left'], early_stopping['best_model'],early_stopping['best_epoch'] = epoch_dev_loss, cfg['patience'], deepcopy(self.state_dict()),epoch
                else: # if the dev loss goes up: patience -= 1
                    early_stopping['patience_left'] -= 1
                if early_stopping['patience_left'] == 0:
                    print('No patience left ... saving model of epoch',early_stopping['best_epoch'])
                    self.load_state_dict(early_stopping['best_model'])
                    break

        # save the final model to file
        self.save_model(self.train_model_dir + '/model.p')
        self.save_model(self.model_dir + '/model.p')


    def get_loss(self, predictions, gt_annotations, loss_functions):

        Losses = {lf: 0 for lf in loss_functions}

        for pred,gt_ann in zip(predictions, gt_annotations):
            for loss_func_str in loss_functions:
                loss_func = eval('losses.'+loss_func_str)
                loss_res = loss_func(pred, gt_ann, self) / len(predictions)
                tmp = Losses[loss_func_str] + loss_res
                Losses[loss_func_str] = tmp

        Losses['L'] = sum(Losses.values())

        return  Losses

    def build_model(self, cfg, data):
        self.d_task = SpanRegressionTaskModel('Dmode',1,self.timeline_task.max_tlv, cfg, data)
        self.d_lower_task = SpanRegressionTaskModel('D-', 0, self.timeline_task.max_tlv, cfg, data)
        self.d_upper_task = SpanRegressionTaskModel('D+', 1, self.timeline_task.max_tlv, cfg, data)
        self.s_task = SpanRegressionTaskModel('Smode',0,self.timeline_task.max_tlv, cfg, data)
        self.s_lower_task = SpanRegressionTaskModel('S-', 0, self.timeline_task.max_tlv, cfg, data)
        self.s_upper_task = SpanRegressionTaskModel('S+', 0, self.timeline_task.max_tlv, cfg, data)
        self.tasks = nn.ModuleDict({model.name:model for model in [self.d_task, self.d_lower_task, self.d_upper_task, self.s_task, self.s_lower_task, self.s_upper_task]})


    def pred_doc(self, doc, loss_functions=[]):
        event_spans = doc.get_sorted_event_spans()

        task_preds = {t:m.pred_spans_in_doc(event_spans, doc) for t,m in self.tasks.items()}

        gt_annotations, raw_preds, pred_annotations = [], [], []
        for i in range(len(event_spans)):

            # --- ground truth ---

            span = event_spans[i]
            gt_ann = doc.event_span_to_event_time_annotations[event_spans[i]]
            gt_annotations.append(gt_ann)

            # --- predict a full event time with bounds ---

            pred = losses.Prediction(span, doc, task_preds['Smode'][i], task_preds['S-'][i], task_preds['S+'][i], task_preds['Dmode'][i], task_preds['D-'][i], task_preds['D+'][i])

            raw_preds.append(pred)
            pred_ann = pred.to_annotation(self)
            pred_annotations.append(pred_ann)

        return {'gt_anns':gt_annotations, 'preds':raw_preds, 'pred_anns':pred_annotations, 'event_spans':event_spans}

    def pred_docs(self,docs, output_dir):
        os.makedirs(output_dir)
        for doc in docs:
            path = output_dir + '/' + doc.id
            copyfile(doc.file_path, path)
            clear_i2b2_file_from_btime_annotations(path)
            pred = self.pred_doc(doc)
            for pred_ann in pred['pred_anns']:
                write_annotation_to_event_in_doc(path,pred_ann)

def load_model(path=None):
    print ('loading model', path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_latest_checkpoint_from_dir(dir):
    latest_path, highest_num = "no checkpoint", 0
    for file_path in glob.glob(dir + "/*.p"):
        num = int(file_path.split('_')[-1][:-2])
        print(file_path, num)
        if num >= highest_num:
            latest_path = file_path
    return latest_path



class Baseline(AbsoluteTimelineModel):

    def __init__(self, cfg, data):
        super(Baseline, self).__init__(cfg, data)

    def pretrain_glove(self, cfg):

        if cfg['glove']:
            glove_model = Glove.load(cfg['glove'])
        else:
            print('pre-training glove...')
            with open(cfg['unlabeled_txt'], 'r') as f:
                lines = []
                for line in f.readlines():
                    lines.append(line.strip().split(' '))
            corpus = Corpus()
            corpus.fit(lines, window=10)
            glove = Glove(no_components=300, learning_rate=0.05)
            glove.fit(corpus.matrix, epochs=30, no_threads=8, verbose=True)
            print('glove fitted (30 epochs)')
            glove.add_dictionary(corpus.dictionary)
            glove.save(self.model_dir + '/glove.model')

        num_words, emb_dim = len(glove_model.dictionary), len(glove_model.word_vectors[0])
        emb_matrix = torch.Tensor(glove_model.word_vectors)
        embs = TextEncoder(docs=[], cfg={'elmo_weights':False, 'elmo_options':False, 'wemb_dim':emb_dim, 'lstm_dim':0, 'linear_reshaper':0, 'dropout':0, 'tied_parameters_regex':"glove",'gpu':cfg['gpu']})
        embs.wembedding = nn.Embedding(num_words + 1, emb_dim)

        embs.windex = glove_model.dictionary
        embs.windex[embs.unk_token] = len(embs.windex)
        mean_vector = torch.mean(emb_matrix, dim=0)
        emb_matrix = torch.cat([emb_matrix, mean_vector.view(1,emb_dim)])
        embs.wembedding.weight = nn.Parameter(emb_matrix)
        return embs

    def build_model(self, cfg, data):
        self.regressor_type = eval(cfg['regressor_type'])
        self.one_minute = self.timeline_task.calendar_duration_to_tlv(CalenderDuration(minutes=1))

        # Duration Network by Vempala et al. (NAACL, 2018)
        self.glove_embedder = self.pretrain_glove(cfg)
        self.additional_embedding = TextEncoder({'elmo_weights':False, 'elmo_options':False, 'wemb_dim':100, 'lstm_dim':0, 'linear_reshaper':0, 'dropout':cfg['dropout'], 'tied_parameters_regex':"NONE",'gpu':cfg['gpu'],'unk_threshold':0},data)
        self.wemb_dim = self.glove_embedder.span_dim + self.additional_embedding.span_dim + 2

        self.d_sent_lstm = nn.LSTM(self.wemb_dim, 200, 1)
        self.d_left_lstm = nn.LSTM(self.wemb_dim, 200, 1)
        self.d_right_lstm = nn.LSTM(self.wemb_dim, 200, 1)
        self.d_event_dim = self.wemb_dim + self.d_sent_lstm.hidden_size + self.d_right_lstm.hidden_size + self.d_left_lstm.hidden_size

        self.regression = self.regressor_type(self.d_event_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg['dropout'], data=data, cfg=self.cfg)
        self.regression_lower = self.regressor_type(self.d_event_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg['dropout'], data=data, cfg=self.cfg)
        self.regression_upper = self.regressor_type(self.d_event_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg['dropout'], data=data, cfg=self.cfg)

        self.dropout = nn.Dropout(cfg['dropout'])
        self.softmax = nn.Softmax()
        self.softplus_nn = nn.Softplus()


    def softplus(self, x):
        return self.one_minute + self.softplus_nn(x)

    def prep_doc(self, doc, max_number_of_events=10000): # used to get re-usable preprocessing steps.
        if not 'duration' in doc.preproc_results:
            doc.preproc_results['duration'] = {}
            for event_span in doc.get_sorted_event_spans():
                event_token_indices = doc.span_to_tokens(event_span,token_index=True)
                sentence = doc.get_sentence_tokens(event_span)
                left_context, event_position, right_context = 1,[[0,1]],0
                for i,tok_index in enumerate(sentence):
                    if tok_index < event_token_indices[0]:
                        left_context += 1
                        event_position.append([0,1])
                    elif tok_index in event_token_indices:
                        event_position.append([1,0])
                        right_context = i + 1
                    else:
                        event_position.append([0,1])
                event_position.append([0,1])

                sentence_string_list = ['<S>'] + [doc.tokens[i] for i in sentence] + ['</S>']
                preproc_dict = {'sentence':sentence_string_list, 'left_context_i':left_context, 'right_context_i':right_context, 'event_position':event_position}
                doc.preproc_results['duration'][event_span] = preproc_dict

    def pred_doc(self, doc, loss_functions=[],max_number_of_events=1000000):
        gt_annotations, raw_preds, pred_annotations = [], [], []
        self.prep_doc(doc)
        event_spans = doc.get_sorted_event_spans()

        for i in range(len(event_spans)):
            event_span = event_spans[i]
            gt_ann = doc.event_span_to_event_time_annotations[event_spans[i]]
            gt_annotations.append(gt_ann)

            sentence_glove_wembs = self.dropout(self.glove_embedder.text_forward(doc.preproc_results['duration'][event_span]['sentence'])['wembs'])
            sentence_additional_wembs = self.dropout(self.additional_embedding.text_forward(doc.preproc_results['duration'][event_span]['sentence'])['wembs'])
            sentence_position_embs = torch.Tensor(doc.preproc_results['duration'][event_span]['event_position']).view(sentence_glove_wembs.size()[0],2).to(self.cfg['gpu'])
            sentence_embedding = torch.cat((sentence_glove_wembs,sentence_position_embs,sentence_additional_wembs),dim=1)

            left_context_length = doc.preproc_results['duration'][event_span]['left_context_i']
            right_context_length = len(doc.preproc_results['duration'][event_span]['sentence']) - doc.preproc_results['duration'][event_span]['right_context_i'] - 1
            left_context_rep = self.d_left_lstm(sentence_embedding[:doc.preproc_results['duration'][event_span]['left_context_i']].view(left_context_length, 1, sentence_embedding.size()[1]))[1][0].view(self.d_left_lstm.hidden_size)
            right_context_rep = self.d_right_lstm(sentence_embedding[doc.preproc_results['duration'][event_span]['right_context_i']+1:].view(right_context_length, 1, sentence_embedding.size()[1]))[1][0].view(self.d_right_lstm.hidden_size)
            sentence_rep = self.d_sent_lstm(sentence_embedding.view(sentence_embedding.size()[0],1,sentence_embedding.size()[1]))[1][0].view(self.d_sent_lstm.hidden_size)
            event_wemb =torch.mean(sentence_embedding[doc.preproc_results['duration'][event_span]['left_context_i']:doc.preproc_results['duration'][event_span]['right_context_i'] + 1],dim=0)
            event_rep = torch.cat((left_context_rep, right_context_rep,sentence_rep,event_wemb),dim=0)

            d_pred = self.softplus(self.regression(event_rep))
            d_lower_sigma = self.softplus(self.regression_lower(event_rep))
            d_upper_sigma = self.softplus(self.regression_upper(event_rep))
            d_lower = d_pred - d_lower_sigma
            d_upper = d_pred + d_upper_sigma


            d_final_prediction, d_lower_prediction, d_upper_prediction = d_pred, d_lower, d_upper
            s_final_prediction, s_lower_prediction, s_upper_prediction = torch.Tensor([1]),torch.Tensor([1]),torch.Tensor([1])

            pred = losses.Prediction(event_span, doc, s_final_prediction[0], s_lower_prediction[0], s_upper_prediction[0], d_final_prediction[0], d_lower_prediction[0], d_upper_prediction[0], intermediate_steps={})
            pred.intermediate_steps['d_dev_left'], pred.intermediate_steps['d_dev_right'] = d_lower_sigma, d_upper_sigma
            raw_preds.append(pred)
            pred_ann = pred.to_annotation(self)
            pred_annotations.append(pred_ann)

        return {'gt_anns':gt_annotations, 'preds':raw_preds, 'pred_anns':pred_annotations, 'event_spans':event_spans}


class ShiftBasedModel(AbsoluteTimelineModel):

    def __init__(self, cfg, data):
        super(ShiftBasedModel, self).__init__(cfg, data)
        self.one_minute = self.timeline_task.calendar_duration_to_tlv(CalenderDuration(minutes=1))
        self.leaky_relu = nn.LeakyReLU(self.one_minute)
        self.relu = nn.ReLU()
        self.softplus_nn = nn.Softplus()


    def softplus(self, x):
        return self.one_minute + self.softplus_nn(x)

    def build_model(self, cfg, data):
        # === Utilities ===
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.regressor_type = eval(cfg['regressor_type'])
        cfg_d = read_cfg(cfg, 'D')
        cfg_s = read_cfg(cfg, 'S')

        # === Duration Network ===
        self.d_text_encoder = SequenceEncoder(cfg_d, data)
        self.d_regression = self.regressor_type(self.d_text_encoder.out_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg_d['dropout'])
        self.d_lower_regression =self.regressor_type(self.d_text_encoder.out_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg_d['dropout'])
        self.d_upper_regression =self.regressor_type(self.d_text_encoder.out_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg_d['dropout'])

        # === Start Point Network ===
        self.s_anchor_text_encoder = SequenceEncoder(cfg_s, data)
        self.s_anchor_classifier = nn.Linear(self.s_anchor_text_encoder.out_dim, 1)
        self.dropout = nn.Dropout(cfg['dropout'])
        self.s_shift_text_encoder = SequenceEncoder(cfg_s, data)
        self.s_shift_regression = self.regressor_type(self.s_shift_text_encoder.out_dim, min=-self.timeline_task.max_tlv, max=self.timeline_task.max_tlv, dropout=cfg_s['dropout'])
        self.s_lower_regression =self.regressor_type(self.s_shift_text_encoder.out_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg_s['dropout'])
        self.s_upper_regression =self.regressor_type(self.s_shift_text_encoder.out_dim, min=self.timeline_task.min_tlv, max=self.timeline_task.max_tlv, dropout=cfg_s['dropout'])
        self.double()

        if self.d_text_encoder.elmo:
            self.d_text_encoder.elmo.float()
            self.s_anchor_text_encoder.elmo.float()
            self.s_shift_text_encoder.elmo.float()

    def pred_doc(self, doc, loss_functions=[], max_number_of_events=1000):
        gt_annotations, raw_preds, pred_annotations = [], [], []
        prepped_doc = self.prep_doc(doc)
        event_spans = doc.get_sorted_event_spans()
        if len(event_spans) > max_number_of_events:
            remove = set(random.sample(event_spans, len(event_spans) - max_number_of_events))
            event_spans = [span for span in event_spans if not span in remove]

        if self.cfg['predict_duration']:
            # Duration prediction from the event and it's local context.
            d_emb = self.d_text_encoder.embed_doc(doc)
            d_enc = self.d_text_encoder(prepped_doc['E_context_tokens'], d_emb)
            #part_size = d_enc.size()[1]
            ds = self.softplus(self.d_regression(d_enc))#[:,:part_size]))
            d_lower_deviations = self.softplus(self.d_lower_regression(d_enc))#[:,part_size:part_size*2]))
            d_upper_deviations = self.softplus(self.d_upper_regression(d_enc))#[:,part_size*2:part_size*3]))
            d_lowers = ds - d_lower_deviations
            d_uppers = ds + d_upper_deviations


        torch.cuda.empty_cache()

        if self.cfg['predict_start']:
            # Start anchor prediction from context between each anchor candidate and the event (candidates are: first times of the text, and the first left and right timex of the event)
            s_anchor_emb = self.s_anchor_text_encoder.embed_doc(doc)
            s_anchor_enc = self.dropout(self.s_anchor_text_encoder(prepped_doc['ETs_contexts'], s_anchor_emb))
            s_anchor_scores = self.s_anchor_classifier(s_anchor_enc)

            anchor_scores_per_event, anchor_values_per_event, best_anchors_et_indices_per_event = [],[],[]
            for i in range(len(event_spans)):
                et_indices = prepped_doc['event_index_to_ETs_indices'][i].to(self.cfg['gpu'])
                anchor_scores = self.softmax(torch.index_select(s_anchor_scores, 0, et_indices).view(len(et_indices)))
                max_score, max_index = anchor_scores.max(0)
                best_anchors_et_indices_per_event.append(et_indices[max_index])
                anchor_scores_per_event.append(anchor_scores)
                del et_indices

            s_shift_emb = self.s_shift_text_encoder.embed_doc(doc)
            s_shift_enc = self.s_shift_text_encoder([prepped_doc['ETs_contexts'][i] for i in best_anchors_et_indices_per_event],s_shift_emb)
            s_shifts = self.s_shift_regression(s_shift_enc) #[:,:part_size])
            s_lower_deviations = self.softplus(self.s_lower_regression(s_shift_enc))
            s_upper_deviations = self.softplus(self.s_upper_regression(s_shift_enc))

        torch.cuda.empty_cache()

        for i in range(len(event_spans)):
            event_span = event_spans[i]
            gt_ann = doc.event_span_to_event_time_annotations[event_spans[i]]
            gt_annotations.append(gt_ann)

            intermediate_steps = {}
            if self.cfg['predict_start']:
                anchor_value = prepped_doc['ETs_values'][best_anchors_et_indices_per_event[i]]
                s =  s_shifts[i] + anchor_value
                s_lower = s - s_lower_deviations[i]
                s_upper = s + s_upper_deviations[i]
                intermediate_steps['s_dev_left'], intermediate_steps['s_dev_right'] = s_lower_deviations[i], s_upper_deviations[i]
                intermediate_steps['anchor_scores'], intermediate_steps['anchor_values'] = anchor_scores_per_event[i], [prepped_doc['ETs_values'][j] for j in prepped_doc['event_index_to_ETs_indices'][i]]
            else:
                s, s_lower, s_upper = torch.Tensor([2.0]).cuda(self.cfg['gpu']), torch.Tensor([1.0]).cuda(self.cfg['gpu']), torch.Tensor([3.0]).cuda(self.cfg['gpu'])

            if self.cfg['predict_duration']:
                d, d_lower,d_upper = ds[i], d_lowers[i], d_uppers[i]
                intermediate_steps['d_dev_left'],intermediate_steps['d_dev_right'] = d_lower_deviations[i], d_upper_deviations[i]
            else:
                d, d_lower, d_upper = torch.Tensor([2.0]).cuda(self.cfg['gpu']), torch.Tensor([1.0]).cuda(self.cfg['gpu']), torch.Tensor([3.0]).cuda(self.cfg['gpu'])

            pred = losses.Prediction(event_span,doc,s[0],s_lower[0],s_upper[0],d[0], d_lower[0], d_upper[0], intermediate_steps=intermediate_steps)
            raw_preds.append(pred)
            pred_ann = pred.to_annotation(self)
            if self.cfg['course_eval']:
                pred_ann = pred_ann.make_course()
            pred_annotations.append(pred_ann)

        return {'gt_anns':gt_annotations, 'preds':raw_preds, 'pred_anns':pred_annotations, 'event_spans':event_spans}



    def get_context_tokens(self,doc, span_1, span_2):
        toks_span_1, toks_span_2 = doc.span_to_tokens(span_1,token_index=True), doc.span_to_tokens(span_2,token_index=True)
        first_token, last_token = min(toks_span_1[0],toks_span_2[0]), max(toks_span_1[-1],toks_span_2[-1])
        context_start, context_end = max(first_token - self.cfg['context_size'],0), min(last_token + self.cfg['context_size'] + 1,len(doc.tokens))
        context_tokens = list(range(context_start, context_end)) #doc.tokens[context_start:context_end]
        if span_1[0] < span_2[0]:
            return context_tokens
        else:
            return list(reversed(context_tokens))

    def prep_doc(self, doc, event_spans=False):
        if not event_spans:
            event_spans = doc.get_sorted_event_spans()

        preproc = {'ETs_values':[], 'ETs_span_pairs':[], 'ETs_contexts':[],'ETd_values':[], 'ETd_span_pairs':[], 'ETd_contexts':[],'E_tokens':[],'E_context_tokens':[],'E_spans':[],'event_index_to_ETs_indices':{}}
        raw_timex_spans_and_values = doc.get_timex_values()

        for j, event_span in enumerate(event_spans):
            preproc['E_context_tokens'].append(doc.n_left_tokens_from_span(event_span, length=self.cfg['context_size'],indices=True) + doc.span_to_tokens(event_span,token_index=True) + doc.n_right_tokens_from_span(event_span, length=self.cfg['context_size'],indices=True))
            preproc['E_tokens'].append(doc.span_to_tokens(event_span))
            preproc['E_spans'].append(event_span)
            preproc['event_index_to_ETs_indices'][j]=[]
            timex_neighbors = doc.get_left_and_right_closest_span_from_span_list(event_span, [sp for sp,_ in raw_timex_spans_and_values['datetimes']])
            for i,(datetime_timex_span, v) in enumerate(raw_timex_spans_and_values['datetimes']):
                if datetime_timex_span in timex_neighbors: # select as candidates: the first timex of the document (admission time), or the first left and first right neighboring timex.
                    context_tokens = self.get_context_tokens(doc, event_span, datetime_timex_span)
                    preproc['ETs_span_pairs'].append((event_span,datetime_timex_span))
                    preproc['ETs_contexts'].append(context_tokens)
                    preproc['ETs_values'].append(self.timeline_task.calendar_point_to_tlv(v[0]))
                    preproc['event_index_to_ETs_indices'][j].append(len(preproc['ETs_contexts']) - 1)
            preproc['event_index_to_ETs_indices'][j] = torch.LongTensor(preproc['event_index_to_ETs_indices'][j])
            for duration_timex_span, v in raw_timex_spans_and_values['durations']:
                if doc.sentence_dist(event_span,duration_timex_span) == 0:
                    context_tokens = self.get_context_tokens(doc, event_span, duration_timex_span)
                    preproc['ETd_span_pairs'].append((event_span,duration_timex_span))
                    preproc['ETd_contexts'].append(context_tokens)
                    preproc['ETd_values'].append(v)
        return preproc


