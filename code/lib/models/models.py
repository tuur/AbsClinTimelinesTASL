from collections import Counter
from functools import reduce
import torch, sys, random, os
import numpy as np
import torch.autograd as autograd
import datetime, time, pickle, random
from baseconvert import base
import lib.losses as losses
from lib.agreement import Evaluation, make_timeline
from lib.i2b2 import  write_annotation_to_event_in_doc
from torch import nn
random.seed(0)
from allennlp.modules.elmo import Elmo, batch_to_ids
from shutil import copyfile
from decimal import Decimal
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import dateutil



import matplotlib.pyplot as plt

from lib.EventTimeAnnotation import CalenderPoint, CalenderDuration


class AbsoluteTimelineModel(nn.Module):

    def __init__(self, cfg, data):
        super(AbsoluteTimelineModel, self).__init__()
        self.cfg = cfg
        if cfg['verbose']:
            print('setting up model...')
        self.model_dir= "./out/" + cfg["experiment_name"]
        self.earliest_date = CalenderPoint(year=cfg['earliest_time']['year'],month=cfg['earliest_time']['month'],day=cfg['earliest_time']['day'],hour=cfg['earliest_time']['hour'],minute=cfg['earliest_time']['minute'])
        self.latest_date= CalenderPoint(year=cfg['latest_time']['year'],month=cfg['latest_time']['month'],day=cfg['latest_time']['day'],hour=cfg['latest_time']['hour'],minute=cfg['latest_time']['minute'])
        self.n_arity = cfg['number_arity']

        self.max_duration_in_minutes = cfg['max_duration'] if cfg['max_duration'] else self.timedelta_to_minute_value(self.latest_date.point - self.earliest_date.point)
        self.num_out_digits = len(base(self.max_duration_in_minutes, 10, self.n_arity))
        self.conversion_matrix = nn.Parameter(torch.FloatTensor([[self.n_arity**digit*i for i in range(self.n_arity)] for digit in range(self.num_out_digits-1,-1,-1)]), requires_grad=False)

        self.unk_token = "_unk_"
        self.windex, self.cindex = self.setup_vocabularies(data)
        self.relu = nn.LeakyReLU()
        self.wembedding = nn.ModuleDict({k:torch.nn.Embedding(len(self.windex), cfg['wemb_dim']) for k in ['S','S-','S+','D','D+','D-']})

        # Text encoding
        self.word_dim = cfg['word_dim']

        self.span_dim, self.char_dim, self.elmo, self.word_dim  =  0, 0, False, cfg['word_dim']
        if cfg['elmo_options'] and cfg['elmo_weights'] :
            self.elmo = Elmo(cfg['elmo_options'], cfg['elmo_weights'], 2, dropout=0)
            self.span_dim += 512

        if cfg['character_dim']:
            self.char_dim = cfg['character_dim']
            self.span_dim += cfg['character_dim'] * 2
            self.character_bilstm = torch.nn.LSTM(input_size=self.char_dim,hidden_size=self.word_dim, bidirectional=True)
            self.character_emb = torch.nn.Embedding(len(self.cindex), self.char_dim)

        if cfg['wemb_dim']:
            self.span_dim += cfg['wemb_dim'] * 2

        self.span_h = nn.ModuleDict({k:nn.Linear(self.span_dim, self.word_dim) for k in ['S','S-','S+','D','D+','D-']})


    def print_model_info(self):
        print('to predict between', self.earliest_date, 'and', self.latest_date, '(', self.max_duration_in_minutes,
              'minute interval)')
        print('using base', self.n_arity, ', resulting in', self.num_out_digits, 'digits to predict, base*digits=',
              self.n_arity * self.num_out_digits)
        #print('windex:', len(self.windex), 'cindex',len(self.cindex))
        print('Total number of parameters:',self.get_number_of_parameters())
        for key,value in dict(self.named_parameters()).items():
            print(key, list(value.size()), reduce(lambda x, y: x * y, value.size()))

    def calender_point_to_minute_value(self, calender_point):
        return self.timedelta_to_minute_value(calender_point.point - self.earliest_date.point)

    def minute_value_to_calender_point(self, minute_value):
        dtime = self.earliest_date.point + datetime.timedelta(minutes=minute_value)
        return CalenderPoint(year=dtime.year, month=dtime.month, day=dtime.day, hour=dtime.hour, minute=dtime.minute)

    def duration_to_minute_value(self, duration):
        return self.timedelta_to_minute_value(duration.duration)

    def minute_value_to_duration(self, minute_value):
        return CalenderDuration(minutes=int(round(minute_value,0)))

    def timedelta_to_minute_value(self, timedelta):
        return int(round(timedelta.total_seconds() // 60,0))

    def get_number_of_parameters(self, trained=True):
        num = 0
        parameter_iterator = self.get_parameters() if not trained else self.trained_parameters()
        for params in parameter_iterator:

                num+=reduce((lambda x, y: x * y), params.shape)
        return num

    def get_parameters(self):
        return self.parameters()


    def trained_parameters(self):
        params = []
        excluded = set(self.cfg['tied_parameters'].split(','))
        for name,param in self.named_parameters():
            add = True
            for excluder in excluded:
                if excluder in name:
                    add = False
            if add:
                params.append(param)
            if self.cfg['verbose']:
                print('train param', name, add)
            #if not 'elmo' in name or ('train_elmo' in self.cfg and self.cfg['train_elmo']==True):
            #    params.append(param)
        return params

    def get_components(self):
        return self.modules()
#        return list(self.digit_layers.values()) + [self.span_h] + ([self.elmo] if self.elmo else [self.character_bilstm, self.character_emb])

    def to_gpu(self, gpu=0):

        if self.cfg['verbose']:
            print('moving model to gpu', gpu)
        self.to(gpu)
        for v in self.windex.values():
            v = v.cuda(gpu)
        for v in self.cindex.values():
            v = v.cuda(gpu)

    def setup_vocabularies(self, data):
        self.word_frequencies = Counter([token for text in data for token in text.tokens])
        self.char_frequencies = Counter([c.lower() if text.lowercased else c for text in data for c in text.text])
        cindex = {w: autograd.Variable(torch.LongTensor([i])) for i, w in enumerate(list(self.char_frequencies.keys()) + [self.unk_token])}
        windex = {w: autograd.Variable(torch.LongTensor([i])) for i, w in enumerate(list(self.word_frequencies.keys()) + [self.unk_token])}
        return windex, cindex

    def fw_text_representation(self, doc):
        encoded_text = {}
        if self.elmo:
            # ELMo
            character_ids = batch_to_ids([doc.tokens]).cuda(self.cfg['gpu'])
            elmo_representations = self.elmo(character_ids)
            encoded_text['elmo'] = elmo_representations['elmo_representations']
        if self.char_dim:
            char_indices = torch.stack([self.cindex[char.lower()] if doc.lowercased else self.cindex[char] for char in doc.text]).cuda(self.cfg['gpu'])
            embedded_characters = self.character_emb(char_indices)
            character_representations = self.character_bilstm(embedded_characters)
            encoded_text['char_lstm'] = character_representations[0]
        return encoded_text

    def fw_get_span_representation(self, span, encoded_text, doc, name):
        token_ids = list(doc.span_to_tokens(span, token_index=True))
        span_repr = torch.Tensor([]).cuda(self.cfg['gpu'])
        if self.elmo:
            first = encoded_text['elmo'][1][0][token_ids[0]]
            last = encoded_text['elmo'][1][0][token_ids[-1]]
            span_repr = torch.cat((span_repr, first,last))
        if self.char_dim:
            left_to_right = encoded_text['char_lstm'][span[1]-1][0][:self.word_dim]
            right_to_left = encoded_text['char_lstm'][span[0]][0][self.word_dim:]
            span_repr = torch.cat((span_repr, left_to_right, right_to_left))

        if self.wembedding:
            wemb = self.wembedding[name](torch.LongTensor([self.windex[doc.tokens[token_ids[0]]] if doc.tokens[token_ids[0]] in self.windex else self.windex[self.unk_token], self.windex[doc.tokens[token_ids[-1]]] if doc.tokens[token_ids[-1]] in self.windex else self.windex[self.unk_token]]).cuda(self.cfg['gpu'])).view(self.cfg['wemb_dim']*2)
            span_repr = torch.cat((span_repr, wemb))

        span_h_repr = self.span_h[name](span_repr) # one hidden layer after the span representation to obtain the desired dimensionality
        return span_h_repr

    def fw_get_attention(self, query, keys, values, head):
        #  currently no attention used
        return query

    def fw_pred_span_digit_probs(self, span_representation, encoded_text, span, doc, name):
        digit_probs = []
        softmax = torch.nn.Softmax(dim=0)
        #sigmoid = torch.nn.Sigmoid()
        for digit in range(self.num_out_digits):
            # apply attention
            digit_representation = self.fw_get_attention(span_representation, encoded_text, encoded_text, name+str(digit))
            # linear layer
            activations = self.digit_layers[name][digit](digit_representation)

            probabilities = softmax(activations) 
            digit_probs.append(probabilities)
        return digit_probs

    def fw_pred_span_value(self, span, encoded_text, doc, name):
        # get the span activations from the character bilstm
        span_representation = self.fw_get_span_representation(span, encoded_text, doc, name)
        # predict digit probabilities
        digit_probs = self.fw_pred_span_digit_probs(span_representation, encoded_text, span, doc, name)
        # convert the digit probabilities to a single float value
        value = self.convert_digit_probs_to_value(digit_probs)

        return value

    def convert_digit_probs_to_value(self, digit_probs):
        if not type(digit_probs) == torch.Tensor:
            digit_probs = torch.cat(digit_probs)
        digit_probs = digit_probs.view(self.num_out_digits, self.n_arity)
        prod = digit_probs * self.conversion_matrix
        summed = sum(sum(prod))
        return summed

    def convert_value_to_vector(self, value):
        converted = base(value, 10, self.n_arity)
        full_length_digit = [0]*(self.num_out_digits-len(converted)) + list(converted)
        digits = []
        for d in range(self.num_out_digits):
            digit_rep = [0] * self.n_arity
            digit_rep[full_length_digit[d]] = 1.0
            digits.append(digit_rep)
        return torch.Tensor(digits)

    def set_train_mode(self):
        for component in self.get_components():
            if hasattr(component, 'train'):
                component.train()

    def set_eval_mode(self):
        for component in self.self.get_components():
            if hasattr(component, 'eval'):
                component.eval()

    def save_model(self, path=None):
        if not path:
            path=self.model_dir + '/model.p'
        print ('saving model', path)
        init_time = time.time()
        with open(path, 'wb') as f:
           pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print('saved t:', round(time.time() - init_time, 2), 's')


    def pred_doc(self, doc, get_losses=[], write_xml_out=False, write_timeline_out=False, eval=False):
        event_spans = sorted(list(doc.event_span_to_event_time_annotations.keys()))

        encoded_text = self.fw_text_representation(doc)


        gt_annotations, raw_preds, pred_annotations = [], [], []
        Ls, Ld = 0, 0
        for i in range(len(event_spans)):

            # --- ground truth ---

            span = event_spans[i]
            gt_ann = doc.event_span_to_event_time_annotations[event_spans[i]]
            gt_annotations.append(gt_ann)

            # --- predict a full event time with bounds ---
            pred = self.pred_span(span, doc, encoded_text)
            raw_preds.append(pred)
            pred_ann = pred.to_annotation(self)
            pred_annotations.append(pred_ann)


            # --- calculating loss ---

            if "L1s" in get_losses:
                Ls += losses.L1_s(pred, gt_ann, self) / len(event_spans)
            if "L1d" in get_losses:
                Ld += losses.L1_d(pred, gt_ann, self) / len(event_spans)
            if 'Llogd' in get_losses:
                Ld += losses.Llog_d(pred, gt_ann, self) / len(event_spans)
            if 'Lprob_d' in get_losses:
                Ld += losses.Lprob_d(pred, gt_ann, self) / len(event_spans)
            if 'Lprob_s' in get_losses:
                Ls += losses.Lprob_s(pred, gt_ann, self) / len(event_spans)
            if 'Ljd' in get_losses:
                Ld += losses.Ljd(pred, gt_ann, self)

        if 'Lt_s' in get_losses:
            Ls += losses.Lt_s(raw_preds, gt_annotations) / self.max_duration_in_minutes


        doc_eval = Evaluation(gt_annotations,pred_annotations) if eval else None

        if write_xml_out:
            copyfile(doc.file_path, write_xml_out)
            for pred_ann in pred_annotations:
                write_annotation_to_event_in_doc(write_xml_out, pred_ann)
        if write_timeline_out:
            event_strings = [doc.text[span[0]:span[1]] for span in event_spans]
            make_timeline(gt_annotations, pred_annotations, event_strings, event_strings,write_timeline_out, max=True)

        return {'Ls':Ls, 'Ld':Ld, 'gt_anns':gt_annotations, 'pred_anns':pred_annotations, 'doc_eval':doc_eval}


    def train_model(self, cfg, data):
        self.to_gpu(cfg['gpu'])

        optimizer = torch.optim.Adam(self.trained_parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], amsgrad=True)
        losses = set([w.replace(' ','') for w in cfg['losses'].split(',')])

        dev_data, train_data = data[:cfg['development_set_size']], data[cfg['development_set_size']:]
        print('starting train:', len(train_data), 'dev:',len(dev_data))
        print('Number of parameters trained:',self.get_number_of_parameters(trained=True))

        if cfg['output_train_pred_xmls']:
            train_pred_file_dir = self.model_dir + '/intermediate_preds/train/'
            dev_pred_file_dir = self.model_dir + '/intermediate_preds/dev/'
            os.makedirs(train_pred_file_dir + '/xml/')
            os.makedirs(train_pred_file_dir + '/viz/')
            os.makedirs(dev_pred_file_dir + '/xml/')
            os.makedirs(dev_pred_file_dir + '/viz/')
        if cfg['checkpoint']:
            checkpoint_dir = self.model_dir +'/checkpoints/'
            os.makedirs(checkpoint_dir)

        epoch_stats = {'Ls':[],'Ld':[],'start_ko':[], 'Pio':[], 'Jd':[], 'dev_Ls':[],'dev_Ld':[],'dev_start_ko':[], 'dev_Pio':[], 'dev_Jd':[]}

        for epoch in range(cfg['max_number_of_epochs']):
            t0 = time.time()
            if cfg['reset_adam'] and epoch % cfg['reset_adam'] == 0:
                print('adam reset')
                optimizer = torch.optim.Adam(self.trained_parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'], amsgrad=True)

            if cfg['checkpoint'] and epoch % cfg['checkpoint'] == 0:
                self.save_model(checkpoint_dir + 'checkpoint_'+str(epoch)+'.p')


            random.shuffle(train_data)
            train_doc_stats = {'Ls': [], 'Ld': [], 'start_ko': [], 'Pio': [], 'Jd':[]}
            self.train()

            for did,doc in enumerate(train_data):

                pred_file_path = train_pred_file_dir + '/xml/e'+ str(epoch) + '-' + doc.id if cfg['output_train_pred_xmls'] and did <  cfg['output_train_pred_xmls'] else False
                write_timeline_path = train_pred_file_dir + '/viz/e' + str(epoch) + '-' + doc.id[:-4] + '-diff.html' if did < cfg['output_train_pred_xmls'] else False

                # make predictions
                predicted_doc = self.pred_doc(doc, get_losses=losses, write_xml_out=pred_file_path, write_timeline_out=write_timeline_path, eval=True)

                doc_loss = predicted_doc['Ld'] + predicted_doc['Ls']
                self.zero_grad()
                doc_loss.backward()
                if cfg['clip']:
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), cfg['clip'])
                    torch.nn.utils.clip_grad.clip_grad_value_(self.parameters(), cfg['clip'])
                optimizer.step()

                train_doc_stats['Ls'].append(predicted_doc['Ls'].data.tolist() if not type(predicted_doc['Ls']) == int else 0)
                train_doc_stats['Ld'].append(predicted_doc['Ld'].data.tolist() if not type(predicted_doc['Ld']) == int else 0)
                train_doc_stats['start_ko'].append(predicted_doc['doc_eval'].metrics['start_ko'])
                train_doc_stats['Pio'].append(predicted_doc['doc_eval'].metrics['Pio'])
                train_doc_stats['Jd'].append(predicted_doc['doc_eval'].metrics['Jd'])



            self.eval()
            dev_doc_stats = {'dev_Ls': [], 'dev_Ld': [], 'dev_start_ko': [], 'dev_Pio': [], 'dev_Jd':[]}
            for did,doc in enumerate(dev_data):
                pred_file_path = dev_pred_file_dir + '/xml/e' + str(epoch) + '-' + doc.id if cfg['output_train_pred_xmls'] and cfg['output_train_pred_xmls'] < did else False
                write_timeline_path = dev_pred_file_dir + '/viz/e' + str(epoch) + '-' + doc.id[:-4] + '-diff.html' if did < cfg['output_train_pred_xmls'] else False
                predicted_doc = self.pred_doc(doc, get_losses=cfg['losses'].split(','), write_xml_out=pred_file_path,
                                              write_timeline_out=write_timeline_path, eval=True)
                dev_doc_stats['dev_Ls'].append(predicted_doc['Ls'].data.tolist() if not type(predicted_doc['Ls']) == int else 0)
                dev_doc_stats['dev_Ld'].append(predicted_doc['Ld'].data.tolist() if not type(predicted_doc['Ld']) == int else 0)
                dev_doc_stats['dev_start_ko'].append(predicted_doc['doc_eval'].metrics['start_ko'])
                dev_doc_stats['dev_Pio'].append(predicted_doc['doc_eval'].metrics['Pio'])
                dev_doc_stats['dev_Jd'].append(predicted_doc['doc_eval'].metrics['Jd'])


            for k, vs in list(train_doc_stats.items())+list(dev_doc_stats.items()):
                epoch_stats[k].append(np.mean(vs))


            plot_data = [go.Scatter(x=np.array(range(cfg['max_number_of_epochs'])), y=np.array(values), mode='lines+markers', name=key)
                         for key, values in epoch_stats.items()]
            py.offline.plot(plot_data, filename=self.model_dir + '/train_stats.html', auto_open=False)

            print('> epoch', epoch + 1, 'Ls', '%.6E' % Decimal(str(epoch_stats['Ls'][-1])), 'Ld', '%.6E' % Decimal(str(epoch_stats['Ld'][-1])),'Jd', '%.6E' % Decimal(str(epoch_stats['Jd'][-1])),'start_ko','%.6E' % Decimal(str(epoch_stats['start_ko'][-1])), 'Pio','%.6E' % Decimal(str(epoch_stats['Pio'][-1])), '(t:',round(time.time()-t0, 1),'s)')





class MinimalATLM(AbsoluteTimelineModel):

    def __init__(self, cfg, data):
        super(MinimalATLM, self).__init__(cfg, data)

       # Linear output weights
        self.digit_layers = nn.ModuleDict({'S':torch.nn.Linear(self.word_dim, 1),
                             'S-': torch.nn.Linear(self.word_dim, 1),
                             'S+': torch.nn.Linear(self.word_dim, 1),
                             'D': torch.nn.Linear(self.word_dim, 1),
                             'D-': torch.nn.Linear(self.word_dim, 1),
                             'D+': torch.nn.Linear(self.word_dim, 1)})

        if cfg['verbose']:
            self.print_model_info()

    def fw_pred_span_value(self, span, encoded_text, doc, name):
        # get the span activations from the character bilstm
        span_representation = self.fw_get_span_representation(span, encoded_text, doc, name)
        # predict digit probabilities

        raw_value = self.digit_layers[name](span_representation)

        value = self.max_duration_in_minutes
        value = raw_value * self.max_duration_in_minutes
        return value[0].clamp(1)


    def pred_span(self, span, doc, encoded_text):
        pred_x_s = self.fw_pred_span_value(span, encoded_text, doc, 'S')
        pred_x_s_lower = pred_x_s - self.fw_pred_span_value(span, encoded_text, doc, 'S-')
        pred_x_s_upper = pred_x_s + self.fw_pred_span_value(span, encoded_text, doc, 'S+')

        pred_x_d = self.fw_pred_span_value(span, encoded_text, doc, 'D')
        pred_x_d_lower = self.fw_pred_span_value(span, encoded_text, doc, 'D-')
        pred_x_d_upper = self.fw_pred_span_value(span, encoded_text, doc, 'D+')

        pred = losses.Prediction(span, doc, pred_x_s, pred_x_s_lower, pred_x_s_upper, pred_x_d, pred_x_d_lower, pred_x_d_upper)

        return pred


class ClassicATLM(AbsoluteTimelineModel):

    def __init__(self, cfg, data):
        super(ClassicATLM, self).__init__(cfg, data)

        # Linear output weights
        self.digit_layers = nn.ModuleDict({'S':nn.ModuleList([torch.nn.Linear(self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]),
                             'S-':nn.ModuleList([torch.nn.Linear(self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]),
                             'S+':nn.ModuleList([torch.nn.Linear(self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]),
                             'D': nn.ModuleList([torch.nn.Linear(self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]),
                             'D-': nn.ModuleList([torch.nn.Linear(self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]),
                             'D+': nn.ModuleList([torch.nn.Linear(self.word_dim, self.n_arity) for digit in range(self.num_out_digits)])})

        if cfg['verbose']:
            self.print_model_info()

    def pred_span(self, span, doc, encoded_text):
        pred_x_s = self.fw_pred_span_value(span, encoded_text, doc, 'S')
        pred_x_s_lower = pred_x_s - self.fw_pred_span_value(span, encoded_text, doc, 'S-')
        pred_x_s_upper = pred_x_s + self.fw_pred_span_value(span, encoded_text, doc, 'S+')

        pred_x_d = self.fw_pred_span_value(span, encoded_text, doc, 'D')
        pred_x_d_lower = self.fw_pred_span_value(span, encoded_text, doc, 'D-')
        pred_x_d_upper = self.fw_pred_span_value(span, encoded_text, doc, 'D+')

        pred = losses.Prediction(span, doc, pred_x_s, pred_x_s_lower, pred_x_s_upper, pred_x_d, pred_x_d_lower, pred_x_d_upper)

        return pred


class ResConnection(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ResConnection, self).__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.intermediate_dim = int(in_dim + out_dim / 2)
        self.fflayer_1 = nn.Linear(in_dim, self.intermediate_dim)
        self.fflayer_2 = nn.Linear(self.intermediate_dim, out_dim)
        self.LRelu = torch.nn.LeakyReLU()

    def forward(self, copy_in, rest_in):
        shape = copy_in.size()
        copy_in = copy_in.flatten()
        input = torch.cat((copy_in, rest_in))
        out_1 = self.LRelu(self.fflayer_1(input))
        out_2 = self.LRelu(self.fflayer_2(out_1))
        return (out_2 + copy_in).view(shape)


class AttATLM(AbsoluteTimelineModel):

    def __init__(self, cfg, data):
        super(AttATLM, self).__init__(cfg, data)

        self.value_dim = self.n_arity * self.num_out_digits
        self.empty_key_value, self.empty_attention_value = nn.Parameter(torch.Tensor(np.zeros(cfg['key_dim'])), requires_grad=False), nn.Parameter(torch.Tensor(np.zeros(self.value_dim)).view(self.num_out_digits, self.n_arity), requires_grad=False)
        self.key_embedders = nn.ModuleDict({k:nn.Linear(self.word_dim, cfg['key_dim']) for k in ['S','S+','S-','D','D+','D-']})
        self.query_embedders = nn.ModuleDict({k:nn.Linear(self.word_dim, cfg['key_dim']) for k in ['S','S+','S-','D','D+','D-']})
        self.event_embedders = nn.ModuleDict({p:nn.Linear(self.word_dim, cfg['key_dim']) for p in ['S','S+','S-','D','D+','D-']})
        self.att_softmax = torch.nn.Softmax(dim=0)

        self.predictors = nn.ModuleDict({p:ResConnection(self.value_dim + cfg['key_dim'], self.n_arity*self.num_out_digits) for p in ['S','S+','S-','D','D+','D-']})


        if cfg['verbose']:
            self.print_model_info()


    def get_value_vector_from_span(self, span, encoded_text, doc):

        v = self.get_value_from_span(span, doc)
        if type(v) == CalenderDuration:  # for durations the value is the TIMEX duration in n-ary encoding
            value = self.convert_value_to_vector(v.in_minutes())
            return value.to(self.cfg['gpu']), 'duration'
        elif type(v) == tuple:
            value = self.convert_value_to_vector(int(np.mean([self.calender_point_to_minute_value(v[0]),
                                                              self.calender_point_to_minute_value(v[1])])))  # for dates and times the value is the middle of the given interval (e.g. 1993 has a n-ary encoded value in the middle of that year)
            return value.to(self.cfg['gpu']), 'datetime'
        else:
            return False



    def fw_text_representation(self, doc):


        # 1) obtain word representations
        encoded_text = super(AttATLM, self).fw_text_representation(doc)

        # 2) obtain representations of temporal expressions, acting as attention keys and values

        # temporal expressions in order of appearance in the text
        timex_s_spans = sorted((doc.span_annotations['type:DATE'] if 'type:DATE' in doc.span_annotations else []) + (doc.span_annotations['type:TIME'] if 'type:TIME' in doc.span_annotations else []), key=lambda x: x[0])
        timex_d_spans = sorted(doc.span_annotations['type:DURATION'] if 'type:DURATION' in doc.span_annotations else [], key=lambda x: x[0])

        keys, values = {}, {}
        for n in ['S','S+','S-']:
            keys_n = [self.empty_key_value]
            values_n = [self.empty_attention_value]
            for span in timex_s_spans + timex_d_spans:
                vt = self.get_value_vector_from_span(span, encoded_text, doc)
                if vt:
                    v, type = vt
                    if type == 'datetime':
                        key = self.key_embedders[n](self.fw_get_span_representation(span, encoded_text, doc,n))
                        keys_n.append(key)
                        values_n.append(v)
            keys[n],values[n] = torch.stack(keys_n), torch.stack(values_n)

        for n in ['D','D+','D-']:
            keys_n = [self.empty_key_value]
            values_n = [self.empty_attention_value]
            for span in timex_s_spans + timex_d_spans:
                vt = self.get_value_vector_from_span(span, encoded_text, doc)
                if vt:
                    v, type = vt
                    if type == 'duration':
                        key = self.key_embedders[n](self.fw_get_span_representation(span, encoded_text, doc,n))
                        keys_n.append(key)
                        values_n.append(v)
            keys[n],values[n] = torch.stack(keys_n), torch.stack(values_n)


        return encoded_text, keys, values


    def attention(self, query, keys, values):

        raw_attention_mask_scores = torch.sum(query * keys, dim=1)
        softmax_attention_mask_s = self.att_softmax(raw_attention_mask_scores)

        attended_values = torch.mul(softmax_attention_mask_s, values.view(len(keys),self.num_out_digits*self.n_arity).t())

        weighted_sum = torch.sum(attended_values, dim=1)

        return weighted_sum.view(self.num_out_digits, self.n_arity)


    def pred_span(self, span, doc, encoded_text):
        #print(doc.text[span[0]:span[1]])
        encoded_elmo_text, keys, values = encoded_text

        preds = {}
        for n in ['S','S+','S-']:
            event_representation_elmo = self.fw_get_span_representation(span, encoded_elmo_text, doc,n)
            event_query_s = self.query_embedders[n](event_representation_elmo)
            attention_value_out_s = self.attention(event_query_s, keys[n], values[n])
            event_embedding = self.event_embedders[n](event_representation_elmo)
            pred = self.predictors[n](copy_in=attention_value_out_s, rest_in=event_embedding)

            preds[n] = self.convert_digit_probs_to_value(pred)


        for n in ['D','D+','D-']:
            event_representation_elmo = self.fw_get_span_representation(span, encoded_elmo_text, doc, n)
            event_query_d = self.query_embedders[n](event_representation_elmo)
            attention_value_out_d = self.attention(event_query_d, keys[n], values[n])
            event_embedding = self.event_embedders[n](event_representation_elmo)
            pred = self.predictors[n](copy_in=attention_value_out_d, rest_in=event_embedding)
            preds[n] = self.convert_digit_probs_to_value(pred)

        pred = losses.Prediction(span, doc, preds['S'], preds['S'] - preds['S-'], preds['S'] + preds['S+'], preds['D'], preds['D-'], preds['D+'])

        return  pred

    def get_value_from_span(self, timex_span, doc):

        timex_labels = doc.reverse_span_annotations[timex_span]

        str = [l for l in timex_labels if 'val:' in l][0][4:]

        if str == '':
            return False

        if str[0].lower() == 'p': # Deals with duration TIMEX
            if str[1].lower() == 't':
                str = str[0] + str[2:]
            if str[-1].lower()=='d':
                duration = CalenderDuration(days=int(float(str[1:-1])))
            elif str[-1].lower()=='h':
                duration = CalenderDuration(hours=int(float(str[1:-1])))
            elif str[-1].lower()=='w':
                duration = CalenderDuration(days=int(7*float(str[1:-1])))
            elif str[-1].lower()=='m':
                duration = CalenderDuration(months=int(float(str[1:-1])))
            elif str[-1].lower() =='y':
                duration = CalenderDuration(years=int(float(str[1:-1])))
            else:
                return False

            return duration

        if len(str) == 4: # Deals with Dates or Times
            start = datetime.datetime.strptime(str, '%Y') # eg 1991
            end = start + dateutil.relativedelta.relativedelta(years=1, minutes=-1)
        elif len(str) == 7:
            start = datetime.datetime.strptime(str, '%Y-%m') # eg 1991-08
            end = start + dateutil.relativedelta.relativedelta(months=1, minutes=-1)
        elif len(str) == 10:
            start = datetime.datetime.strptime(str, '%Y-%m-%d') # eg 2001-03-20
            end = start + dateutil.relativedelta.relativedelta(days=1, minutes=-1)
        elif len(str) == 16:
            str = [l for l in timex_labels if 'val:' in l][0][4:] # 1997-04-01t17:20
            start = datetime.datetime.strptime(str, '%Y-%m-%dt%H:%M')
            end = start + dateutil.relativedelta.relativedelta(minutes=1)

        else:
            return False
        return (CalenderPoint(year=start.year, month=start.month, day=start.day, hour=start.hour, minute=start.minute), CalenderPoint(year=end.year, month=end.month, day=end.day, hour=end.hour, minute=end.minute))


class MinimalAttATLM(AttATLM):
    def __init__(self, cfg, data):
        super(MinimalAttATLM, self).__init__(cfg, data)

        # attention values are different, namely only 1 value
        self.value_dim = 1
        self.empty_attention_value =  nn.Parameter(torch.Tensor(np.zeros(self.value_dim)), requires_grad=False)
        self.predictors = nn.ModuleDict({p:ResConnection(self.value_dim + cfg['key_dim'], self.value_dim) for p in ['S','S+','S-','D','D+','D-']})

    def convert_value_to_vector(self, value):
        return torch.Tensor([value])


    def attention(self, query, keys, values):

        raw_attention_mask_scores = torch.sum(query * keys, dim=1)
        softmax_attention_mask_s = self.att_softmax(raw_attention_mask_scores)


        attended_values = torch.mul(softmax_attention_mask_s, values.view(len(keys),self.value_dim).t())

        weighted_sum = torch.sum(attended_values, dim=1)


        return weighted_sum.view(self.value_dim)

    def pred_span(self, span, doc, encoded_text):

        encoded_elmo_text, keys, values = encoded_text

        preds = {}
        for n in ['S','S+','S-']:
            event_representation_elmo = self.fw_get_span_representation(span, encoded_elmo_text, doc,n)
            event_query_s = self.query_embedders[n](event_representation_elmo)
            attention_value_out_s = self.attention(event_query_s, keys[n], values[n])
            event_embedding = self.event_embedders[n](event_representation_elmo)
            pred = self.predictors[n](copy_in=attention_value_out_s, rest_in=event_embedding)

            preds[n] = pred[0].clamp(1)

        for n in ['D','D+','D-']:
            event_representation_elmo = self.fw_get_span_representation(span, encoded_elmo_text, doc, n)
            event_query_d = self.query_embedders[n](event_representation_elmo)
            attention_value_out_d = self.attention(event_query_d, keys[n], values[n])
            event_embedding = self.event_embedders[n](event_representation_elmo)
            pred = self.predictors[n](copy_in=attention_value_out_d, rest_in=event_embedding)
            preds[n] = pred[0]

        pred = losses.Prediction(span, doc, preds['S'], preds['S'] - preds['S-'], preds['S'] + preds['S+'], preds['D'].clamp(1), preds['D-'].clamp(0),preds['D+'].clamp(0))

        return  pred



def load_model(path=None):
    print ('loading model', path)
    with open(path, 'rb') as f:
        return pickle.load(f)
