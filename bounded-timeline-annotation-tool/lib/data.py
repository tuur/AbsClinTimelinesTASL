# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:59:21 2017

@author: tuur
"""

from __future__ import print_function
import nltk, glob, os, re, time
from nltk import sent_tokenize
import nltk.tag.stanford
from nltk.tag.stanford import StanfordPOSTagger as POSTagger
nltk.data.path.append("venv/nltk_data")


#_default_pos_model_ = './stanford-postagger/models/english-bidirectional-distsim.tagger'

#current_dir = os.getcwd()
#_path_to_jar = current_dir + '/stanford-postagger/stanford-postagger.jar'
#default_pos_model = POSTagger(model_filename=current_dir + '/stanford-postagger/models/english-left3words-distsim.tagger', path_to_jar=_path_to_jar)
#caseless_pos_model = POSTagger(model_filename=current_dir + '/stanford-postagger/models/english-caseless-left3words-distsim.tagger', path_to_jar=_path_to_jar)



class Text(object):
	
	def __init__(self, text, span_annotations={}, span_pair_annotations={}, id=0, lowercase=True, pos=False, transitive_closure=[], conflate_digits=True, measure_speed=False, file_path=None):
		if measure_speed:
			t0 = time.time()
		self.span_annotations = span_annotations
		self.reverse_span_annotations = reverse_dict_list(self.span_annotations)
		self.span_pair_annotations = span_pair_annotations			
		self.reverse_span_pair_annotations = reverse_dict_list(self.span_pair_annotations)
		if len(transitive_closure) > 0:
			for label in transitive_closure:
				self.take_transitive_closure(label)
		self.text = text
		self.tokens, self.spans = tokenize(text, lowercase=lowercase, conflate_digits=conflate_digits)
		self.span_starts, self.span_ends = {s:i for (i,(s,e)) in enumerate(self.spans)}, {e:i for (i,(s,e)) in enumerate(self.spans)}
		self.id = id
		self.file_path = file_path
		self.pos = []
		self.lowercased = lowercase
		if pos:
			self.pos = self.parse_pos()
		else:
			self.pos = ['no_pos' for tok in self.tokens]
		
		
		self.entity_spans, self.entity_indices, self.entity_tokens, self.entities = self.get_entity_spans()
		self.vocabulary, self.pos_vocabulary = self.get_vocabs()
		#print(self.span_annotations)		
		self.sentence_boundaries, self.character_index_to_sentence_index = get_sentence_boundaries(text)
		self.paragraph_boundaries, self.character_index_to_paragraph_index = get_paragraph_boundaries(text)
		if 'CONTAINS' in self.span_pair_annotations and len(self.span_pair_annotations['CONTAINS']) > 0 and not 'clinic' in self.id:
			print('YES', self.id, len(self.span_pair_annotations['CONTAINS']))
		if measure_speed:
			print(self.id,'read t:',time.time()-t0,'s', 'words:', len(self.tokens),'w/s:', float(len(self.tokens)) / (time.time()-t0))

	def get_entity_spans(self):
		event_spans = self.span_annotations['EType:EVENT'] if 'EType:EVENT' in self.span_annotations else []
		timex3_spans = self.span_annotations['EType:TIMEX3'] if 'EType:TIMEX3' in self.span_annotations else []
		entity_spans = list(sorted([sp for sp in event_spans + timex3_spans],key=lambda x:x[0]))
		entity_indices = {sp:i for i,sp in enumerate(entity_spans)}
		entity_token_list = [token for span in entity_spans for token in self.span_to_tokens(span)]

		event_token_indices = set([i for event_span in event_spans for i in self.span_to_tokens(event_span, token_index=True)])
		timex3_token_indices = set([i for timex3_span in timex3_spans for i in self.span_to_tokens(timex3_span, token_index=True)])
		token_level_entity_labels = ['EType:EVENT' if i in event_token_indices else 'EType:TIMEX3' if i in timex3_token_indices else 'O' for  i,t in enumerate(self.tokens)]

		#print(len(entity_indices), len(entity_token_list))
		return entity_spans, entity_indices, entity_token_list, token_level_entity_labels



	def filter_distant_pairs(self, max_sentence_distance=1):
		removed = 0
		for label in self.span_pair_annotations:
			for a1,a2 in self.span_pair_annotations[label]:
				if self.sentence_dist(a1,a2) > max_sentence_distance and not (a1==(0,0) or a2==(0,0)):
					self.span_pair_annotations[label].remove((a1,a2))
					del self.reverse_span_pair_annotations[(a1,a2)]
					removed+=1
		print('removed',removed, 'TLinks (sentence dist filter)')
		

	def ning_closure(self, mult =False):
		abbrev = {'BEFORE':'B','AFTER':'A','INCLUDES':'I','IS_INCLUDED':'II', 'SIMULTANEOUS':'S'}
		rev_abbrev = {abbr:lab for lab,abbr in abbrev.items()}
		triplets = {('B','B'):['B'], ('B','S'):['B'],('A','A'):['A'], ('A','S'):['A'], ('I','I'):['I'], ('I','S'):['I'], ('II','II'):['II'], ('II','S'):['II'],('S','B'):['B'],('S','A'):['A'],('S','I'):['I'],('S','II'):['II'],('S','S'):['S']}
		multiple_out_triplets = {('B','I'):['B','I'],('B','II'):['B','II'],('A','I'):['A','I'],('A','II'):['A','II'],('I','B'):['B','I'],('I','A'):['A','I'],('II','B'):['B','II'],('II','A'):['A','II']}
		c=0
		if mult:
			triplets = triplets.update(multiple_out_triplets)
		new_annotations = {l:set() for l in abbrev.keys()}
		for r1 in self.reverse_span_pair_annotations:
			for r2 in self.reverse_span_pair_annotations:
				if r1[1]==r2[0] and not r1==r2 and self.sentence_dist(r1[0],r1[1]):
					l1s = [abbrev[l] for l in self.reverse_span_pair_annotations[r1] if l in abbrev]
					l2s = [abbrev[l] for l in self.reverse_span_pair_annotations[r2] if l in abbrev]
					
					for l1 in l1s:
						for l2 in l2s:
							if (l1,l2) in triplets:								
								new_pair = (r1[0],r2[1])
								for new_l in triplets[(l1,l2)]:
									new_l = rev_abbrev[new_l]
									if not new_pair in self.span_pair_annotations[new_l]:
										new_annotations[new_l].add(new_pair)
										c+=1
		
		
		self.update_annotations(span_pair_update=new_annotations)
		print('added by closure:', c)
				
				
				
			
		
		
		
	
	def take_transitive_closure(self, span_pair_label):
		# update reverse and normal annotations
		added_relations = set([])
		
		if not span_pair_label in self.span_pair_annotations:
			return added_relations
		for span_a, span_b in self.span_pair_annotations[span_pair_label]:
			if span_a == span_b:
				continue
			for span_c, span_d in self.span_pair_annotations[span_pair_label]:
				if span_b == span_c and not (span_a == span_d or span_a == span_c):
					new_relation = (span_a, span_d) 
					if new_relation in added_relations:
						continue
					if not new_relation in self.reverse_span_pair_annotations:
						self.span_pair_annotations[span_pair_label].append(new_relation)
						self.reverse_span_annotations[new_relation] = [span_pair_label]
						added_relations.add(new_relation)

					elif not span_pair_label in self.reverse_span_pair_annotations[(span_a, span_d)]:
						self.span_pair_annotations[span_pair_label].append(new_relation)
						self.reverse_span_pair_annotations[new_relation].append(span_pair_label)
						added_relations.add(new_relation)
		print('added', len(added_relations), 'x', span_pair_label, '(transitive closure)')
		return added_relations						
	
	def sim_rel_extension(self):
		if not 'SIMULTANEOUS' in self.span_pair_annotations:
			return

		# each relation to span_a should also go to span_b
		count = 0
		
		span_can_reach_span = {}
		span_is_reached_by_span = {}

		for span_a, span_b in self.reverse_span_pair_annotations:
			rels = [r for r in self.reverse_span_pair_annotations[(span_a, span_b)] if r!='SIMULTANEOUS']
			if not span_a in span_can_reach_span:
				span_can_reach_span[span_a] = {}
			if not span_b in span_can_reach_span[span_a]:
				span_can_reach_span[span_a][span_b] = []			
			if not span_b in span_is_reached_by_span:
				span_is_reached_by_span[span_b] = {}
			if not span_a in span_is_reached_by_span[span_b]:
				span_is_reached_by_span[span_b][span_a] = []
				
			span_can_reach_span[span_a][span_b] += rels
			span_is_reached_by_span[span_b][span_a] += rels

		for span_a, span_b in self.span_pair_annotations['SIMULTANEOUS']: 
			
			if span_a in span_can_reach_span: # A-sim->B and A-r->C then B-r->C
				for span_c in span_can_reach_span[span_a]:
					for rel in span_can_reach_span[span_a][span_c]:
						if not (span_b, span_c) in self.span_pair_annotations[rel]:
							self.span_pair_annotations[rel].append((span_b, span_c))
							if not (span_b, span_c) in self.reverse_span_annotations:
								self.reverse_span_annotations[(span_b, span_c)]=[]
							self.reverse_span_annotations[(span_b, span_c)].append(rel)
							count += 1

			if span_a in span_is_reached_by_span: # A-sim->B and C-r->A then C-r->B
				for span_c in span_is_reached_by_span[span_a]:
					for rel in span_is_reached_by_span[span_a][span_c]:
						if not (span_c, span_b) in self.span_pair_annotations[rel]:
							self.span_pair_annotations[rel].append((span_c, span_b))
							if not (span_c, span_b) in self.reverse_span_annotations:
								self.reverse_span_annotations[(span_c, span_b)]=[]
							self.reverse_span_annotations[(span_c, span_b)].append(rel)
							count += 1

		for span_b, span_a in self.span_pair_annotations['SIMULTANEOUS']: 
			
			if span_a in span_can_reach_span: # now also inherit from B to A
				for span_c in span_can_reach_span[span_a]:
					for rel in span_can_reach_span[span_a][span_c]:
						if not (span_b, span_c) in self.span_pair_annotations[rel]:
							self.span_pair_annotations[rel].append((span_b, span_c))
							if not (span_b, span_c) in self.reverse_span_annotations:
								self.reverse_span_annotations[(span_b, span_c)]=[]
							self.reverse_span_annotations[(span_b, span_c)].append(rel)
							count += 1

			if span_a in span_is_reached_by_span: # A-sim->B and C-r->A then C-r->B
				for span_c in span_is_reached_by_span[span_a]:
					for rel in span_is_reached_by_span[span_a][span_c]:
						if not (span_c, span_b) in self.span_pair_annotations[rel]:
							self.span_pair_annotations[rel].append((span_c, span_b))
							if not (span_c, span_b) in self.reverse_span_annotations:
								self.reverse_span_annotations[(span_c, span_b)]=[]
							self.reverse_span_annotations[(span_c, span_b)].append(rel)
							count += 1

		print('SIM Extension added', count, 'relations')
	

		

	def parse_pos(self):
		print('POS Tagging...')
		if self.lowercased:
			tagger = caseless_pos_model
		else:
			tagger = default_pos_model
		
		exceptions = {'\n':'NEWLINE','\t':'TAB', '':'NOTHING'} # not done by stanford

		selected_pos =  [pos for (word,pos) in tagger.tag(self.tokens)]

		final_pos = []
		i=0
		for tok in self.tokens:
			if tok in exceptions:
				final_pos.append(exceptions[tok])
			else:
				final_pos.append(selected_pos[i])
				i+=1

		return final_pos
		
	def get_vocabs(self):

		vocab = {}
		pos_vocab = {}
		for i,token in enumerate(self.tokens):
			pos = self.pos[i] 
			if token in vocab:
				vocab[token].append(i)
			else:
				vocab[token] = [i]
				
			if pos in pos_vocab:
				pos_vocab[pos].append(i)
			else:
				pos_vocab[pos] = [i]
		return vocab, pos_vocab

	def sentence_dist(self, span_a, span_b):
		sia = self.get_sentence_index(span_a)
		sib = self.get_sentence_index(span_b)
		return abs(sia-sib)

	def get_sentence_index(self, span):
		return self.character_index_to_sentence_index[span[0]]
		
	def get_paragraph_index(self, span):
		return self.character_index_to_paragraph_index[span[0]]

	def spans_lie_within_one_sentence(self, span1, span2):
		return self.get_sentence_index(span1) == self.get_sentence_index(span2)
	
	def spans_lie_within_one_paragraph(self,span1, span2):
		return self.get_paragraph_index(span1) == self.get_paragraph_index(span2)
		
	def get_span_labels_by_regex(self, regex):
		labels = set()
		for lab in self.span_annotations:
			if re.search(regex, lab):
				labels.add(lab)
		return labels

	def update_annotations(self, span_update=None, span_pair_update=None):
		if span_update:
			for label in span_update:
				self.span_annotations[label] = span_update[label]
		self.reverse_span_annotations = reverse_dict_list(self.span_annotations)
		if span_pair_update:
			for label in span_pair_update:
				self.span_pair_annotations[label] = span_pair_update[label] 
		self.reverse_span_pair_annotations = reverse_dict_list(self.span_pair_annotations)
		

	def token_distance(self, span_1, span_2):
		first, second = min(span_1[-1],span_2[-1]), max(span_1[0],span_2[0])

		if not second in self.span_starts or not first in self.span_ends:
			return None

		return self.span_starts[second] - self.span_ends[first]

	def tokens_inbetween(self, first, second, pos=False):
		end, start = first[-1], second[0]
		if not end in self.span_ends:
			end = self.get_closest_viable_token_end(end)
		if not start in self.span_starts:
			start = self.get_closest_viable_token_start(start)
		first_index, last_index = self.span_ends[end], self.span_starts[start]
		
		if pos:
			return self.pos[first_index+1:last_index]
		else:
			return self.tokens[first_index+1:last_index]

	def span_to_tokens(self, span, pos=False, token_index=False):
		start, end = span
		if not start in self.span_starts:
			start = self.get_closest_viable_token_start(start)
		if not end in self.span_ends:
			end = self.get_closest_viable_token_end(end)
			
		first_index, last_index = self.span_starts[start], self.span_ends[end]
		if token_index:
			return range(first_index, last_index + 1)
		if pos:
			return self.pos[first_index:last_index+1]
		else:
			return self.tokens[first_index:last_index+1]

	def n_left_tokens_from_span(self, span, length, pos=False):
		start, _ = span
		if not start in self.span_starts:
			start = self.get_closest_viable_token_start(start)
		first_index = self.span_starts[start]
		if pos:
			return self.pos[max(0, first_index-length):first_index]	
		else:
			return self.tokens[max(0, first_index-length):first_index]		
	
	def n_right_tokens_from_span(self, span, length, pos=False):
		_, end = span
		if not end in self.span_ends:
			end = self.get_closest_viable_token_end(end)
		last_index = self.span_ends[end]
		if pos:
			return self.pos[last_index+1:min(last_index + length + 1, len(self.tokens))]
		else:
			return self.tokens[last_index+1:min(last_index + length + 1, len(self.tokens))]

	def span_to_string(self, span):
		return self.text[span[0]:span[1]]
	
	def get_closest_viable_token_start(self, init): # find closest character index of a viable token
		i = 0
		while init + i < len(self.text) and init - i >= 0:
			if init + i in self.span_starts:
				return init + i
			if init - i in self.span_starts:
				return init - i
			i += 1

	def get_closest_viable_token_end(self, init): # find closest character index of a viable token
		i = 0
		while init + i < len(self.text) and init - i >= 0:
			if init + i in self.span_ends:
				return init + i
			if init - i in self.span_ends:
				return init - i	
			i += 1

	def write_to_brat(self, directory, span_label_filter=None, span_pair_label_filter=None):
		if not os.path.isdir(directory):
			os.mkdir(directory)
		print ('to brat:', self.id)
		
		# writing txt file
		with open(directory + '/' + self.id + '.txt', 'w') as f:
			f.write(self.text)
		
		entity_index = 1
		relation_index= 1
		span_to_brat_index = {}
			
		with open(directory + '/' + self.id + '.ann', 'w') as f:
			for label in self.span_annotations:
				if not span_label_filter or re.search(span_label_filter, label):
					
					for span in self.span_annotations[label]:
						f.write('T' + str(entity_index) + '\t' + label.replace(' ','_') + ' ' + str(span[0]) + ' ' + str(span[1]) + '\t' + self.text[span[0]:span[1]] + '\n')
						span_to_brat_index[span] = 'T' + str(entity_index)
						entity_index += 1
						
						
			for label in self.span_pair_annotations:
				if not span_pair_label_filter or re.search(span_pair_label_filter, label):
					for span1, span2 in self.span_pair_annotations[label]:
						#print(label, span1, span2, span_to_brat_index[span1], span_to_brat_index[span2])
						if not span2 == (0,0):
							f.write('R' + str(relation_index) + '\t' + label.replace(' ','_') + ' Arg1:' + span_to_brat_index[span1] + ' Arg2:' + span_to_brat_index[span2] +  '\n')
							#R10	CONTAINS Arg1:E59 Arg2:E62
							relation_index += 1


	def add_PREP_features(self):
		if self.pos ==[]: # prepositions on the left and right (up to next entity)
			print('WARNING: PREP feature cannot find POS')
			return
		
		prep_token_indices = [i for i,p in enumerate(self.pos) if p=='IN']
		prep_left_right_pairs = [(None,prep_token_indices[0])] + list(zip(prep_token_indices, prep_token_indices[1:])) + [(prep_token_indices[-1], None)]

		prep_pair_index = 0
		for i in range(len(self.tokens)):
			if i == prep_left_right_pairs[prep_pair_index][1]:
				prep_pair_index += 1
				
			prep_left_index, prep_right_index = prep_left_right_pairs[prep_pair_index]
			#print('----------',self.tokens[i])
			prep_tok_left = self.tokens[prep_left_index] if type(prep_left_index)==int else 'NONE'
			prep_tok_right = self.tokens[prep_right_index] if type(prep_right_index)==int else 'NONE'

			for f in ['PREPL:'+prep_tok_left, 'PREPR:'+prep_tok_right]:
				if not f in self.span_annotations:
					self.span_annotations[f] = []
				if not self.spans[i] in self.reverse_span_annotations:
					self.reverse_span_annotations[self.spans[i]] = []
				self.span_annotations[f].append(self.spans[i])
				self.reverse_span_annotations[self.spans[i]].append(f)
		

	def add_VERB_features	(self):
		if self.pos ==[]:
			print('WARNING: VERB feature cannot find POS')
			return
		verb_token_indices = [i for i,p in enumerate(self.pos) if p[0]=='V']
		verb_left_right_pairs = [(None,verb_token_indices[0])] + list(zip(verb_token_indices, verb_token_indices[1:])) + [(verb_token_indices[-1], None)]
		
		#VINDEX, VPOS, VTOKEN
		verb_pair_index = 0
		for i in range(len(self.tokens)):
			if i == verb_left_right_pairs[verb_pair_index][1]:
				verb_pair_index += 1
				
			v_left_index, v_right_index = verb_left_right_pairs[verb_pair_index]
			#print('----------',self.tokens[i])
			v_tok_left = self.tokens[v_left_index] if type(v_left_index)==int else 'NONE'
			v_tok_right = self.tokens[v_right_index] if type(v_right_index)==int else 'NONE'
			v_pos_left = self.pos[v_left_index] if type(v_left_index)==int else 'NONE'
			v_pos_right = self.pos[v_right_index] if type(v_right_index)==int else 'NONE'

			vindex = round((verb_pair_index)/(len(verb_left_right_pairs)),1)
			#print(vindex, v_tok_left ,v_pos_left,  v_tok_right, v_pos_right)
			
			vindex = ['VINDEX:'+str(vindex)]
			vpos = ['VPOSL:'+v_pos_left, 'VPOSR:'+v_pos_right]
			vtoken = ['VTOKL:'+v_tok_left, 'VTOKR:'+v_tok_right]
			
			for f in vindex + vpos + vtoken:
				if not f in self.span_annotations:
					self.span_annotations[f] = []
				if not self.spans[i] in self.reverse_span_annotations:
					self.reverse_span_annotations[self.spans[i]] = []
				self.span_annotations[f].append(self.spans[i])
				self.reverse_span_annotations[self.spans[i]].append(f)

		

def transform_to_unidirectonal_relations_text(text):
	new_annotations = transform_to_unidirectonal_relations(text.span_pair_annotations)
	text.span_pair_annotations=new_annotations	
	text.reverse_span_pair_annotations = reverse_dict_list(text.span_pair_annotations)
	return text
	
def transform_to_unidirectonal_relations(span_pair_annotations):
		'''Adds an extra label for each relation pair label, indicating that the relation is in the other direction (w.r.t. word order).
		e.g. If "The dog was walked by John"  would give WALKS(John, dog) we instead introduce the label WALKS_BY, and label it as WALKS_BY(dog, John) instead.
		This allows for unidrectional candidate generation (at the cost of adding an extra label).
		'''
		counter=0
		new_annotations = {}
		for label in span_pair_annotations:
			reverse_label = label + '_INVERSE'
			new_annotations[reverse_label] = []
			new_annotations[label] = []

			for (span_a1, span_a2) in span_pair_annotations[label]:
				if span_a1[0] > span_a2[0]:
					counter+=1
					new_annotations[reverse_label].append((span_a2, span_a1))
				else:
					new_annotations[label].append((span_a1, span_a2))
		return new_annotations			

def transform_to_bidirectonal_relations_text(text):
	new_annotations = transform_to_bidirectonal_relations(text.span_pair_annotations)
	text.span_pair_annotations=new_annotations	
	text.reverse_span_pair_annotations = reverse_dict_list(text.span_pair_annotations)
	return text

	
def transform_to_bidirectonal_relations(span_pair_annotations):
		counter=0
		new_annotations = {}
		for label in span_pair_annotations:
			if label[-8:]=='_INVERSE':
				original_label = label[:-8]
				if not original_label in new_annotations:
					new_annotations[original_label] = []
				
				for (span_a2, span_a1) in span_pair_annotations[label]:
					new_annotations[original_label].append((span_a1, span_a2))
					counter+=1
			else:
				if not label in new_annotations:
					new_annotations[label]=span_pair_annotations[label]
				else:
					new_annotations[label]+=span_pair_annotations[label]
					
		return new_annotations



def read_text_files(directory, lowercase=False, pos=False):
	print('\nReading txt files from', directory)
	texts = []
	for txt_file in glob.glob(directory +"/*.txt"):
		print(txt_file)
		with open(txt_file, 'r') as f:
			text = Text(f.read(), pos=pos, lowercase=lowercase)
			texts.append(text)
	return texts
			

def get_sentence_boundaries(text):
	sents = sent_tokenize(text)
	sent_id = 0
	j=0
	correction=0
	boundaries = []
	character_index_to_sentence_index = {}
	for i, char in enumerate(text):
		character_index_to_sentence_index[i] = len(boundaries)
		if j==len(sents[sent_id]):
			boundaries.append(i)
			j=0
			sent_id+=1
		if sent_id >= len(sents):
			sent_id = sent_id-1
		sent_char=sents[sent_id][j]
		
		if char==sent_char:
			j+=1
			continue
		if char!=sent_char:
			correction+=1
	return boundaries, character_index_to_sentence_index

def get_paragraph_boundaries(text):
	boundaries = []
	character_index_to_paragraph_index = {}
	prev = ''
	for i, char in enumerate(text):
		character_index_to_paragraph_index[i] = len(boundaries)
		if char == '\n' and prev =='\n':
			boundaries.append(i)
		prev=char
	return boundaries, character_index_to_paragraph_index

def reverse_dict_list(d):
	d_new = {}
	for k,l in d.items():
		for v in l:
			if v in d_new and not k in d_new[v]:
				d_new[v].append(k)
			if not v in d_new:
				d_new[v] = [k]					
	return d_new	


	



def tokenize(text, lowercase=True, conflate_digits=True):
	inclusive_splitters = set([',','.','/','\\','"','\n','=','+','-',';',':','(',')','!','?',"'",'<','>','%','&','$','*','|','[',']','{','}'])
	exclusive_splitters = set([' ','\t'])
	tokens = []
	spans = []
	mem = ""
	start = 0
	for i,char in enumerate(text):	
		if char in inclusive_splitters:
			if mem!="":
				tokens.append(text[start:i])
				spans.append((start,i))
				mem = ""
			tokens.append(text[i:i+1])
			spans.append((i,i+1))
			start = i+1
		elif char in exclusive_splitters:
			if mem!="":
				tokens.append(text[start:i])
				spans.append((start,i))
				mem = ""
				start = i+1
			else:
				start = i+1
		else:
			mem += char
	#print mem, start
	if not mem=="":
		tokens.append(mem)
	
	if lowercase:
		tokens = [t.lower() for t in tokens]
	if conflate_digits:
		tokens = [re.sub('\d', '5', t) for t in tokens]
	return tokens, spans	



class Logger(object):
	
	def __init__(self, stream, file_name, log_prefix=None):
		self.log = open(file_name, "w")
		self.stream = stream
		if log_prefix:
			self.log.write('LOG_PREFIX:\n' + log_prefix + '\n\nLOG:\n')

	def write(self, message):
		self.stream.write(message)
		self.log.write(message)  

	def write_to_file(self, message):
		self.log.write(message)
		
	def flush(self):
		pass		
