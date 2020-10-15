from collections import Counter
from functools import reduce
import torch
import numpy as np
import torch.autograd as autograd
import datetime, time, pickle, random
from baseconvert import base

import matplotlib.pyplot as plt



class AbsoluteTimelineModel(object):

    def __init__(self, earliest_date, latest_date, data=[], model_dir="model", char_dim=10, word_dim=25, dropout=0.25, n_arity=3):
        print('setting up model...')
        self.model_dir=model_dir
        self.earliest_date, self.latest_date=earliest_date, latest_date
        self.n_arity = n_arity
        self.max_duration_in_minutes = (self.latest_date.point - self.earliest_date.point).days * 24 * 60
        self.num_out_digits = len(base(self.max_duration_in_minutes, 10, self.n_arity))
        print('to predict between',self.earliest_date, 'and',self.latest_date, '(',self.max_duration_in_minutes,'minute interval)')
        print('using base',self.n_arity,', resulting in', self.num_out_digits,'digits to predict, base*digits=',self.n_arity*self.num_out_digits)

        self.unk_token = "_unk_"
        self.windex, self.cindex = self.setup_vocabularies(data)
        self.char_dim, self.word_dim = char_dim, word_dim
        print('windex:', len(self.windex), 'cindex',len(self.cindex))

        # character LSTM encoding the text
        self.character_bilstm = torch.nn.LSTM(input_size=self.char_dim,hidden_size=self.word_dim, bidirectional=True)
        self.character_emb = torch.nn.Embedding(len(self.cindex), char_dim)

        # Linear output weights
        self.start_digit_layers = [torch.nn.Linear(2*self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]
        self.duration_digit_layers = [torch.nn.Linear(2*self.word_dim, self.n_arity) for digit in range(self.num_out_digits)]

        print('number of parameters:',self.get_number_of_parameters())

    def get_number_of_parameters(self):
        num = 0
        for params in self.get_parameters():
            for p in params:
                num+=reduce((lambda x, y: x * y), p.shape)
        return num

    def get_parameters(self):
        params = []
        for component in self.get_components():
            params.append(component.parameters())
        return params

    def get_components(self):
        return [self.character_bilstm, self.character_emb] + self.start_digit_layers + self.duration_digit_layers

    def to_gpu(self, gpu=0):

        for component in self.get_components():
            component.cuda(gpu)
        for k,v in self.windex.items():
            self.windex[k] = v.cuda(gpu)
        for k,v in self.cindex.items():
            self.cindex[k] = v.cuda(gpu)

    def setup_vocabularies(self, data):
        self.word_frequencies = Counter([token for text in data for token in text.tokens])
        self.char_frequencies = Counter([c.lower() if text.lowercased else c for text in data for c in text.text])
        cindex = {w: autograd.Variable(torch.from_numpy(np.array([i]))) for i, w in enumerate(list(self.char_frequencies.keys()) + [self.unk_token])}
        windex = {w: autograd.Variable(torch.from_numpy(np.array([i]))) for i, w in enumerate(list(self.word_frequencies.keys()) + [self.unk_token])}
        return windex, cindex

    def fw_char_representations(self, doc):
        char_indices = torch.stack([self.cindex[char.lower()] if doc.lowercased else self.cindex[char] for char in doc.text])
        embedded_characters = self.character_emb(char_indices)
        character_representations = self.character_bilstm(embedded_characters)
        return character_representations[0]

    def fw_predict_nary_repr_for_spans_in_doc(self, spans, doc):
        encoded_text = self.fw_char_representations(doc)
        n_ary_span_preds = [self.fw_pred_span_probs(span, encoded_text, doc) for span in spans]
        return n_ary_span_preds

    def fw_get_span_representation(self, span, encoded_text, doc):
        left_to_right = encoded_text[span[1]-1][0][:self.word_dim]
        right_to_left = encoded_text[span[0]][0][self.word_dim:]
        span_repr = torch.cat((left_to_right, right_to_left))
        return span_repr

    def fw_get_attention(self, query, keys, values, head):
        # TODO: currently no attention used
        return query

    def fw_pred_span_start_probs(self, span_representation, encoded_text, span, doc):
        digit_probs = []
        softmax = torch.nn.Softmax(dim=0)
        for digit in range(self.num_out_digits):
            # apply attention
            digit_representation = self.fw_get_attention(span_representation, encoded_text, encoded_text, 'S'+str(digit))
            # linear layer
            activations = self.start_digit_layers[digit](digit_representation)
            # softmax
            probabilities = softmax(activations)
            digit_probs.append(probabilities)
        return digit_probs

    def fw_pred_span_duration_probs(self, span_representation, encoded_text, span, doc):
        digit_probs = []
        softmax = torch.nn.Softmax(dim=0)
        for digit in range(self.num_out_digits):
            # apply attention
            digit_representation = self.fw_get_attention(span_representation, encoded_text, encoded_text, 'D' + str(digit))
            # linear layer
            activations = self.duration_digit_layers[digit](digit_representation)
            # softmax
            probabilities = softmax(activations)
            digit_probs.append(probabilities)
        return digit_probs

    def fw_pred_span_probs(self, span, encoded_text, doc):
        # get the span activations from the character bilstm
        span_representation = self.fw_get_span_representation(span, encoded_text, doc)
        # predict start digit probabilities
        start_digit_probs = self.fw_pred_span_start_probs(span_representation, encoded_text, span, doc)
        # predict duration digit probabilities
        duration_digit_probs = self.fw_pred_span_duration_probs(span_representation, encoded_text, span, doc)
        return start_digit_probs, duration_digit_probs

    def start_and_end_from_digit_probs(self, start_digit_probs, duration_digit_probs):

        # convert probs to digits
        start_digits = ''.join([str(index.data.item()) for _,index in [i.max(0) for i in start_digit_probs]])
        duration_digits = ''.join([str(index.data.item()) for _,index in [i.max(0) for i in duration_digit_probs]])

        # convert the base-n number to a base-10 number representing the number of minutes
        start_in_minutes = int(base(start_digits,self.n_arity, 10, string=True))
        duration_in_minutes = int(base(duration_digits, self.n_arity, 10, string=True))

        # convert to datetime.timedelta to allow calculations with calender dates
        start_timedelta = datetime.timedelta(minutes=start_in_minutes)
        duration_timedelta = datetime.timedelta(minutes=duration_in_minutes)

        # get start and end date from start and duration
        start_date = self.earliest_date.point + start_timedelta
        end_date = start_date + duration_timedelta

        return start_date, end_date

    def predict_events_in_doc(self, doc):
        event_spans = doc.span_annotations['EType:EVENT']
        print('predicting',len(event_spans),'events in',doc.id)
        nary_task_predictions = self.fw_predict_nary_repr_for_spans_in_doc(event_spans, doc)

        start_and_endings = []
        for event_span, (start_probs, duration_probs) in zip(event_spans,nary_task_predictions):
            start, end = self.start_and_end_from_digit_probs(start_probs, duration_probs)
            start_and_endings.append((start, end))
        return event_spans, start_and_endings

    def set_train_mode(self):
        for component in self.get_components():
            if hasattr(component, 'train'):
                component.train()

    def set_eval_mode(self):
        for component in self.self.get_components():
            if hasattr(component, 'eval'):
                component.eval()

    def save_model(self, path):
        print ('saving model', path)
        init_time = time.time()
        with open(path, 'wb') as f:
           pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print('saved t:', round(time.time() - init_time, 2), 's')

    def load_model(self, path):
        print ('loading model', path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def train(self, data, num_epochs=5, patience=10, dev_size=1, gpu=0):
        print('moving model to gpu')
        self.to_gpu(gpu)

        dev_data, train_data = data[:dev_size], data[dev_size:]

        for epoch in range(num_epochs):
            print('> epoch', epoch + 1)
            random.shuffle(train_data)
            for doc in train_data:

                event_spans = doc.span_annotations['EType:EVENT']
                nary_task_predictions = self.fw_predict_nary_repr_for_spans_in_doc(event_spans, doc)

                e0 = event_spans[0]
                s0, d0 = nary_task_predictions[0]


                s0[0][0]=.9
                s0[0][1]=0.1
                s0[0][2]=0.0
                s0[1][0]=.1
                s0[1][1]=0.9
                s0[1][2]=0.0
                s0[2][0] = 1.0
                s0[2][1] = 0.0
                s0[2][2] = 0.0
                s0[3][0] = 0.333
                s0[3][1] = 0.333
                s0[3][2] = 0.333

                print(e0, doc.text[e0[0]:e0[1]])
                print(len(s0), s0)
                nums = range(0, self.max_duration_in_minutes, 100000)
                ps = []
                for i in nums:
                    nary_enc = base(i,10, self.n_arity)
                    nary_enc = (self.num_out_digits-len(nary_enc))*(0,) + nary_enc
                    p = reduce(lambda x,y: x*y, [s0[j][v] for j,v in enumerate(nary_enc)], 1)
                    ps.append(p)

                plt.plot(nums, ps)
                #plt.axis([0, 6, 0, 20])
                plt.show()
                exit()
                #preds = self.predict_events_in_doc(doc)






#data, num_epochs=5, max_docs=None, viz_inbetween=False, verbose=0,save_checkpoints=None, eval_on=None, batch_size=32, temporal_awareness_ref_dir=None, clip=1.0, pred_relations=None, patience=100, loss_func=None, pointwise_loss=None,tune_margin=1, checkpoint_interval=1000,timex3_dur_loss=False, reset_optimizer=Non

