from __future__ import print_function
from lib.data import Text, reverse_dict_list
import glob, codecs, re, os, shutil
import xml.etree.ElementTree as ET
import datetime
from bs4 import BeautifulSoup
from xml.dom import minidom

def extend_annotations_verb_clause(existing_documents, extension_file_path, replace_annotations=False, verbose=0):
	new_annotations = {doc.id:[] for doc in existing_documents}
	with open(extension_file_path, 'r') as f:
		for line in f.readlines()[:-1]:
			if line[0]!='#':
				doc_id, a1, a2, label = line.rstrip().split()
				a1, a2 = a1.replace('tmx','tid:t').replace('ei','eiid:ei'), a2.replace('tmx','tid:t').replace('ei','eiid:ei')
				new_annotations[doc_id].append((a1,a2,label))
	skipped, done = 0, 0
	for doc in existing_documents:
		for (a1,a2,label) in new_annotations[doc.id]:
			if (not a1 in doc.span_annotations) or (not a2 in doc.span_annotations):
				skipped += 1
				continue
			done+=1
			span_a1, span_a2 = doc.span_annotations[a1][0], doc.span_annotations[a2][0]
			pair = span_a1, span_a2
			if not label in doc.span_pair_annotations:
				doc.span_pair_annotations[label] = []
				
			if pair in doc.reverse_span_pair_annotations:
				if replace_annotations:
					current_label = doc.reverse_span_pair_annotations[pair][0]
					doc.span_pair_annotations[current_label].remove(pair)
					doc.span_pair_annotations[label].append(pair)	
			else:
				doc.span_pair_annotations[label].append(pair)					
		doc.reverse_span_pair_annotations = reverse_dict_list(doc.span_pair_annotations)
	print('skipped', skipped, 'added', done)
	return existing_documents

def extend_annotations_timebankdense(existing_documents, extension_file_path, replace_annotations=False, filter_documents=False, include_vague=False, verbose=0):
	timebank_dense_abbreviations = {'a':'AFTER', 'b':'BEFORE', 'i':'INCLUDES', 's':'SIMULTANEOUS', 'ii':'IS_INCLUDED', 'v':'VAGUE'}
	entity_pair_labels_per_document = {}
	total_count = 0
	with codecs.open(extension_file_path, 'r') as f:
		for line in f.readlines():
			Filename, A1, A2, label = line.strip().split('\t')
			#print Filename, A1, A2, label
			if not Filename in entity_pair_labels_per_document:
				entity_pair_labels_per_document[Filename] = {}
			
			if not label in entity_pair_labels_per_document[Filename]:
				entity_pair_labels_per_document[Filename][label] = []
			entity_pair_labels_per_document[Filename][label].append((A1,A2))
	to_be_removed_docs = []
	for doc in existing_documents:
		
		if doc.id in entity_pair_labels_per_document:
			count = 0
			if replace_annotations == True:
				doc.span_pair_annotations = {}
				doc.reverse_span_pair_annotations = {}
				
			for label in entity_pair_labels_per_document[doc.id]:
				full_label = timebank_dense_abbreviations[label]
				if not (include_vague) and full_label == 'VAGUE':
					continue
				if not full_label in doc.span_pair_annotations:
					doc.span_pair_annotations[full_label] = []
				for a1, a2 in entity_pair_labels_per_document[doc.id][label]:
					full_a1 = 'tid:' + a1 if 't' in a1 else 'eid:' + a1
					full_a2 = 'tid:' + a2 if 't' in a2 else 'eid:' + a2
					span_a1 = doc.span_annotations[full_a1][0]
					span_a2 = doc.span_annotations[full_a2][0]
					print(span_a1,span_a2,full_label)
					if not (span_a1, span_a2) in doc.reverse_span_annotations: # Only add relations if there is no relation yet!
						doc.span_pair_annotations[full_label].append((span_a1, span_a2))
						count+=1
					else:
						print((span_a1, span_a2), 'already present')
			if verbose:
				print (count,'\t', doc.id)
			total_count += count
		elif filter_documents:
			to_be_removed_docs.append(doc)
		else:
			if verbose:
				print ('0 \t', doc.id)					
		doc.update_annotations()
		
	if filter_documents:
		if verbose:
			print ('removed docs:', len(to_be_removed_docs))
		for doc in to_be_removed_docs:
			if doc in existing_documents:
				existing_documents.remove(doc)	
	
	print ('added relations:', total_count)	
	return existing_documents
	






def extend_annotations_reimers(existing_documents, extension_file_path):
	entity_labels_per_document = {}
	
	# read annotations
	with codecs.open(extension_file_path, 'r') as f:
		for line in f.readlines():
			Filename, SentenceNumber, TokenNumber, TagName, TagID, InstanceID, AttributeName, AttributeValue = line.strip().split('\t')
			if not Filename in entity_labels_per_document:
				entity_labels_per_document[Filename] = {}
			entity_labels_per_document[Filename][TagID] = AttributeValue
			
	# assign annotations
	for doc in existing_documents:
		if doc.id in entity_labels_per_document:
			count = 0 
			for eid, e_time in entity_labels_per_document[doc.id].items():
				for span in doc.span_annotations['eid:'+eid]:
					#print '\t', span, eid, e_time
					if not 'Reimersetal:'+e_time in doc.span_annotations:
						doc.span_annotations['Reimersetal:'+e_time]=[]
					doc.span_annotations['Reimersetal:'+e_time].append(span)
					count+=1
			doc.reverse_span_annotations = reverse_dict_list(doc.span_annotations)
			print (count,'\t', doc.id)
			
		else:
			print ('0 \t', doc.id)
		
	return existing_documents				
					

def write_timebank_folder(docs, out_dir, verbose=1):
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)
	for doc in docs:
		doc.update_annotations()
		with open(out_dir + '/' + doc.id + '.tml', 'w') as f:
			# TimeML
			permitted_attributes = set(['eid', 'eiid','value','tid', 'temporalFunction','functionInDocument','type','EType','class','anchorTimeID','beginPoint'])
			doc_xml = ET.Element('TimeML')
			doc_xml.attrib['xmlns:xsi']='http://www.w3.org/2001/XMLSchema-instance'
			doc_xml.attrib['xsi:noNamespaceSchemaLocation']='http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd'
			
			# > DOCID
			docid = ET.SubElement(doc_xml,'DOCID')
			docid.text = doc.id

			# > DCT
			dct = ET.SubElement(doc_xml,'DCT')
			dct = ET.SubElement(dct,'TIMEX3')

			span_to_id = {'eid':{},'tid':{}, 'eiid':{}}
			eid_to_eiid = {}			
			temp_value = '00:00:00'
			for ann in doc.reverse_span_annotations[(0,0)]:
				split = ann.split(':')
				key, value = split[0],':'.join(split[1:])
				if not key in permitted_attributes:
					continue
				dct.attrib[key]=value
				if key == 'tid':
					span_to_id['tid'][(0,0)]=value

				elif key=='value':
					temp_value=value
			dct.text = temp_value
			# > EXTRAINFO
			extrainfo = ET.SubElement(doc_xml,'EXTRAINFO')
			extrainfo.text="..."
			# > TEXT
			text_xml = ET.SubElement(doc_xml,'TEXT')
			entity_labels = doc.get_span_labels_by_regex('EType')
			span_starts = {span[0]:span for elab in entity_labels for span in doc.span_annotations[elab]}
			text_str, char_i = "", 0

			while(char_i < len(doc.text)):
				char = doc.text[char_i]
				to_be_printed, length = char, 1
				if char_i in span_starts:
					start, end = span_starts[char_i]
					if not (start,end) == (0,0):		
						length = end-start
						annotations = doc.reverse_span_annotations[(start, end)]
						if [ann for ann in annotations if ann.split(':')[0]=='EType'] != [] :
							
							annotation_type = [ann for ann in annotations if ann.split(':')[0]=='EType'][0].split(':')[1]
							#print(annotation_type, char_i, doc.text[start:end], length, annotations)
							xml_ann = ET.Element(annotation_type)
							xml_ann.text = doc.text[start:end]
							ann_keys = set()
							for ann in annotations:
								split = ann.split(':')
								key, value = split[0],':'.join(split[1:])
								if not key in permitted_attributes:
									continue
								ann_keys.add(key)
								if not (key=='EType' or key=='eiid'):
									xml_ann.attrib[key]=value	
								if key=='eid' or key=='tid' or key=='eiid':
									span_to_id[key][(start,end)]=value
														
							if not ('eid' in ann_keys) and annotation_type=='EVENT':
								print('ERROR: no eid!',doc.id, xml_ann.text)
								exit()
							elif not('tid' in ann_keys) and annotation_type=='TIMEX3':
								print('ERROR: no tid!',doc.id, xml_ann.text)
								exit()
							to_be_printed = ET.tostring(xml_ann,encoding="unicode")
				text_str += to_be_printed
				char_i += length
			text_xml.text=text_str
			
			lastextrainfo = ET.SubElement(doc_xml,'LASTEXTRAINFO')
			lastextrainfo.text = doc.id

			for span in span_to_id['eid']:
				if span in span_to_id['eiid']:
					eid_to_eiid[span_to_id['eid'][span]] = span_to_id['eiid'][span]
			
			for eid in eid_to_eiid:
				makeinstance = ET.SubElement(doc_xml,'MAKEINSTANCE')
				makeinstance.attrib = {'eventID':str(eid), 'eiid':str(eid_to_eiid[eid])}


			lid = 1
			for span_pair, rels in doc.reverse_span_pair_annotations.items():
				for rel in rels:
					sp1, sp2 = span_pair
					id1 = span_to_id['eiid'][sp1] if sp1 in span_to_id['eiid'] else span_to_id['tid'][sp1]
					id2 = span_to_id['eiid'][sp2] if sp2 in span_to_id['eiid'] else span_to_id['tid'][sp2]
					tlink = ET.SubElement(doc_xml,'TLINK')
					tlink.attrib = {'lid':'l'+str(lid), 'relType':rel}
					lid+=1
					if 't'in id1:
						tlink.attrib['timeID']=id1
					else:
						tlink.attrib['eventInstanceID']=id1
						
					if 't' in id2:
						tlink.attrib['relatedToTime']=id2
					else:
						tlink.attrib['relatedToEventInstance'] = id2
					#print(span_pair, rel, id1, id2)
					
			#<TIMEX3 tid="t0" type="TIME" value="1998-02-12T01:58:00" temporalFunction="false" functionInDocument="CREATION_TIME">02/12/1998 01:58:00</TIMEX3>
			
			#doc_xml_string = ET.tostring(doc_xml, encoding='UTF-8')
			#print (type(doc_xml_string))
			doc_xml_string = minidom.parseString(ET.tostring(doc_xml,encoding="unicode").replace('\t', '')).toprettyxml(indent = "", newl='\n')
			#print (doc_xml_string)
			if verbose > 0:
				print('written',doc.id)
			
			f.write(doc_xml_string.replace('&lt;','<').replace('&gt;','>').replace('&quot;', '"'))
		

def read_timebank_folder(folder, verbose=1, conflate_digits=False, pos=False, lowercase=True):
	docs = []
	total_tlinks = 0
	for file_path in glob.glob(folder + "*.tml"):
		with codecs.open(file_path, 'r') as f:
			num_events, num_timex3, num_tlinks = 0,0,0
			xml_str = f.read()
			xmlSoup = BeautifulSoup(xml_str, 'xml')
			text =  xmlSoup.find_all('TEXT')[0]
			raw_txt = ""
			
		
			# Read Entity Spans
			entity_labels = ['EType:EVENT', 'EType:TIMEX3']	
			label_to_spans = {l:[] for l in entity_labels}
			
			# add DCT
			dct_span=(0,0)
			for attr,value in xmlSoup.find('DCT').find('TIMEX3').attrs.items() :
				
				label = attr+':'+value
				
				if not label in label_to_spans:
					label_to_spans[label]=[]
				label_to_spans[label].append(dct_span)	

			for content in text.contents:
				start = len(raw_txt)
				end = start + len(content.string)
				span = (start, end)
				
				if content.name == 'EVENT':
					num_events += 1
					eid=content.attrs['eid']
					label_to_spans['EType:'+content.name].append(span)		

					for attr,value in content.attrs.items():
						label = attr+':'+value 
						if not label in label_to_spans:
							label_to_spans[label]=[]
						label_to_spans[label].append(span)
#					label_to_spans[content.name].append(span)
					
				elif content.name == 'TIMEX3':
					num_timex3 += 1
					eid=content.attrs['tid']
					label_to_spans['EType:'+content.name].append(span)		
					for attr,value in content.attrs.items():
						label = attr+':'+value 
						if not label in label_to_spans:
							label_to_spans[label]=[]
						label_to_spans[label].append(span)

				raw_txt += content.string

			# Adding EIIDs and other info to Entities
			for instance in xmlSoup.find_all('MAKEINSTANCE'):
				eiid = instance.attrs['eiid']
				eid = instance.attrs['eventID']
				span = label_to_spans['eid:'+eid][0]
				label_to_spans['eiid:'+eiid] = [span]
				#print('---', span)
				for attr in instance.attrs:
					if not (attr == 'eiid' or attr == 'eventID'):
						lab= attr + ':' + instance.attrs[attr]
						#print(lab)
						if not lab in label_to_spans:
							label_to_spans[lab] = []
						label_to_spans[lab].append(span)
						#print(attr)
			
			# Add TLINK Annotations
			#link_labels = ['BEFORE','AFTER','INCLUDES','IS_INCLUDED','DURING','DURING_INV','SIMULTANEOUS', 'IAFTER', 'IBEFORE', 'IDENTITY','BEGINS', 'ENDS', 'BEGUN_BY', 'ENDED_BY']
			label_to_span_pairs = {} #{l:[] for l in link_labels}
			for tlink in xmlSoup.find_all('TLINK') + xmlSoup.find_all('ALINK') :
				num_tlinks += 1
				#print tlink
				link_type = tlink.attrs['relType']
				if len(link_type) < 1:
					continue
				e1 = tlink.attrs['eventInstanceID'] if 'eventInstanceID' in tlink.attrs else tlink.attrs['timeID']
				e2 = tlink.attrs['relatedToEventInstance'] if 'relatedToEventInstance' in tlink.attrs else tlink.attrs['relatedToTime']
				sp_e1 = label_to_spans['eiid:'+e1][0] if 'eiid:'+e1 in label_to_spans else label_to_spans['tid:'+e1][0]
				sp_e2 = label_to_spans['eiid:'+e2][0] if 'eiid:'+e2 in label_to_spans else label_to_spans['tid:'+e2][0]
				if not link_type in label_to_span_pairs:
					label_to_span_pairs[link_type] = []
				label_to_span_pairs[link_type].append((sp_e1, sp_e2))

			
	
			# remove possible duplicates:
			for lab in label_to_span_pairs:
				label_to_span_pairs[lab] = list(set(label_to_span_pairs[lab]))
	
			doc_id = xmlSoup.find('DOCID').text			
			text = Text(raw_txt, span_annotations=label_to_spans, span_pair_annotations=label_to_span_pairs, id=doc_id,conflate_digits=conflate_digits, pos=pos, lowercase=lowercase)
			docs.append(text)
			if verbose:
				print(doc_id, '\tevents:', num_events, 'timex3:',num_timex3, 'tlinks:', num_tlinks)
		total_tlinks += num_tlinks
	return docs

def simplify_relations(doc, simplification=1):
	if simplification == 1: # DURING, DURING_INV, SIMULTANEOUS, IDENTITY --> SIMULTANEOUS
		conversion = {'DURING':'SIMULTANEOUS', 'IDENTITY':'SIMULTANEOUS', 'ENDED_BY':'INCLUDES', 'ENDS':'IS_INCLUDED', 'BEGUN_BY':'INCLUDES', 'BEGINS':'IS_INCLUDED', 'CONTINUES':'AFTER', 'INITIATES':'IS_INCLUDED', 'TERMINATES':'IS_INCLUDED', 'IBEFORE':'BEFORE','IAFTER':'AFTER'}
		argument_handling = lambda a1,a2 :(a1,a2)
	elif simplification == 2:	# CONVERSION to TimeBank Dense Labels (in a similar way as Ning et al., 2017)
		conversion = {'DURING':'IS_INCLUDED', 'IDENTITY':'SIMULTANEOUS', 'ENDED_BY':'INCLUDES', 'ENDS':'IS_INCLUDED', 'BEGUN_BY':'INCLUDES', 'BEGINS':'IS_INCLUDED', 'CONTINUES':'AFTER', 'INITIATES':'IS_INCLUDED', 'TERMINATES':'IS_INCLUDED', 'IBEFORE':'BEFORE','IAFTER':'AFTER'}
		#conversion = {'DURING':'IS_INCLUDED',, 'IDENTITY':'SIMULTANEOUS', 'IBEFORE':'BEFORE', 'IAFTER':'AFTER', 'BEGINS':'IS_INCLUDED', 'BEGUN_BY':'INCLUDES', 'ENDS':'IS_INCLUDED', 'ENDED_BY':'INCLUDES'}		
		argument_handling = lambda a1,a2 :(a1,a2)
	elif simplification == 3: # resolve reverse relations
		conversion = {'AFTER':'BEFORE', 'IS_INCLUDED':'INCLUDES', 'BEGUN_BY':'BEGINS', 'ENDED_BY':'ENDS', 'IAFTER':'IBEFORE', 'DURING_INV':'DURING'}				
		argument_handling = lambda a1,a2 :(a2,a1)
	elif simplification == 4:
		conversion = {'IDENTITY':'SIMULTANEOUS'}
		argument_handling = lambda a1,a2 :(a1,a2)	
	elif simplification == 5: # CATENA Conversion
		conversion = {'BEGINS':'BEFORE','ENDED_BY':'BEFORE','BEGUN_BY':'AFTER','ENDS':'AFTER','DURING_INV':'SIMULTANEOUS','DURING':'SIMULTANEOUS', 'IDENTITY':'SIMULTANEOUS','CONTINUES':'AFTER', 'INITIATES':'IS_INCLUDED', 'TERMINATES':'IS_INCLUDED', 'IBEFORE':'BEFORE','IAFTER':'AFTER'}
		argument_handling = lambda a1,a2 :(a1,a2)			
	elif simplification == 6: # Ning et al (2017) Conversion
		conversion = {'DURING':'IS_INCLUDED', 'IDENTITY':'SIMULTANEOUS', 'ENDED_BY':'INCLUDES', 'ENDS':'IS_INCLUDED', 'BEGUN_BY':'INCLUDES', 'BEGINS':'IS_INCLUDED', 'CONTINUES':'AFTER', 'INITIATES':'IS_INCLUDED', 'TERMINATES':'IS_INCLUDED', 'IBEFORE':'BEFORE','IAFTER':'AFTER'}
		argument_handling = lambda a1,a2 :(a1,a2)	

		
	new_annotations = {}
	for rel_type in doc.span_pair_annotations:
		if rel_type in conversion:
			target_rel_type = conversion[rel_type]
			if target_rel_type in doc.span_pair_annotations:
				doc.span_pair_annotations[target_rel_type] += doc.span_pair_annotations[rel_type]
			else:
				new_annotations[target_rel_type] = doc.span_pair_annotations[rel_type]
	
	for rel_type in conversion:
		if rel_type in doc.span_pair_annotations: 	
			del doc.span_pair_annotations[rel_type]
	for target_rel_type in conversion.values():
		if not rel_type in doc.span_pair_annotations and target_rel_type in new_annotations:
			doc.span_pair_annotations[target_rel_type] = new_annotations[target_rel_type]		
			
	doc.reverse_span_pair_annotations = reverse_dict_list(doc.span_pair_annotations)			
	return doc		






def get_normalized_values(doc):
	raw_values = doc.get_span_labels_by_regex('value:\d\d\d\d*')
	normalized_values = {}
	for raw_value in raw_values:
		
		# normalized format: year-quarter-month-weeknr-day-time
		
		month_to_quarter = {'01':'1', '02':'1', '03':'1', '04':'2', '05':'2', '06':'2', '07':'3', '08':'3', '09':'3', '10':'4', '11':'4', '12':'4', }
		
		normalized = ''

		
	
		yearmatch = re.search(r'value:(?P<year>\d\d\d\d)$', raw_value)
		monthmatch = re.search(r'value:(?P<year>\d\d\d\d)-(?P<month>\d\d)$', raw_value)
		datematch = re.search(r'value:(?P<year>\d\d\d\d)-(?P<month>\d\d)-(?P<day>\d\d)$', raw_value)
		quartermatch = re.search(r'value:(?P<year>\d\d\d\d)-Q(?P<quarter>\d)$', raw_value)
		timematch = re.search(r'value:(?P<year>\d\d\d\d)-(?P<month>\d\d)-(?P<day>\d\d)(?P<time>T.*)$', raw_value)
		weekmatch = re.search(r'value:(?P<year>\d\d\d\d)-W(?P<week>\d\d*)$', raw_value)

		if yearmatch:
			normalized = yearmatch.group('year')
		elif monthmatch:
			normalized = monthmatch.group('year') + '-' + month_to_quarter[monthmatch.group('month')] + '-' + monthmatch.group('month')
		elif datematch:
			weeknr = datetime.date(int(datematch.group('year')), int(datematch.group('month')), int(datematch.group('day'))).isocalendar()[1]
			normalized = datematch.group('year') + '-' + month_to_quarter[datematch.group('month')] + '-' + datematch.group('month') + '-' + str(weeknr) + '-' + datematch.group('day')
		elif quartermatch:
			normalized = quartermatch.group('year') + '-' + quartermatch.group('quarter')
		elif timematch:
			weeknr = datetime.date(int(timematch.group('year')), int(timematch.group('month')), int(timematch.group('day'))).isocalendar()[1]
			normalized = timematch.group('year') + '-' + month_to_quarter[timematch.group('month')] + '-' + timematch.group('month') + '-' + str(weeknr) + '-' + timematch.group('day') + '-' + timematch.group('time')
		elif weekmatch:
			normalized = weekmatch.group('year') + '-XX-XX-' + weekmatch.group('week')
		
		
		# normalization still misses a few instances (no real problem)

		if normalized != '':
			normalized_values[raw_value] = normalized	
	return normalized_values

def get_temp_rel(d1, d2): # getting relation between two normalized temporal values
		shortest, longest, rev = (d1,d2, False) if len(d1) < len(d2) else (d2, d1, True)
		for i in range(len(shortest)):
			charv1, charv2 = shortest[i], longest[i]
			if charv1 == charv2 or charv1 == 'X' or charv2 == 'X':
				continue
			if charv1 < charv2:
				return 'BEFORE' if not rev else 'AFTER'
			elif charv1 > charv2:
				return 'AFTER' if not rev else 'BEFORE'
		return 'INCLUDES' if not rev else 'IS_INCLUDED'
	
def extend_tlinks_with_timex3_values(doc):
	normalized_values = get_normalized_values(doc)
		#print(raw_value, '-------> ', normalized)
		
	relations = {}	
	for v1 in normalized_values.values():
		for v2 in normalized_values.values():
			if not v1==v2:
				rel = get_temp_rel(v1,v2)
				relations[(v1,v2)] = rel
	
	
	spans = [(sp,lab) for lab in normalized_values for sp in doc.span_annotations[lab]]
	for i,(span_1,lab_1) in enumerate(spans):
		for (span_2, lab_2) in spans[i:]:
			if not lab_1 == lab_2:
				rel = relations[(normalized_values[lab_1],normalized_values[lab_2])]
				#print(span_1, span_2, lab_1, lab_2, rel)
				if not rel in doc.span_pair_annotations:
					doc.span_pair_annotations[rel] = []
				doc.span_pair_annotations[rel].append((span_1, span_2))
				if not (span_1, span_2) in doc.reverse_span_pair_annotations:
					doc.reverse_span_pair_annotations[(span_1, span_2)] = []
				doc.reverse_span_pair_annotations[(span_1, span_2)].append(rel)


	for l in ['value:FUTURE_REF', 'value:PRESENT_REF','PAST_REF']:
		if l in doc.span_annotations:
			dct_rel = None
			for span in doc.span_annotations[l]:
				if l=='value:FUTURE_REF':
					dct_rel = 'AFTER'
				elif l == 'value:PRESENT_REF':
					dct_rel = 'IS_INCLUDED'
				elif l == 'PAST_REF':
					dct_rel = 'BEFORE'
				
				if dct_rel:
					span_pair = (span,(0,0))
					if not dct_rel in doc.span_pair_annotations:
						doc.span_pair_annotations[dct_rel] = []
					if not span_pair in doc.reverse_span_pair_annotations:
						doc.reverse_span_pair_annotations[span_pair] = []
					doc.span_pair_annotations[dct_rel].append(span_pair)
					doc.reverse_span_pair_annotations[span_pair].append(dct_rel)

	
	return
		

def add_TIF_features(doc):
	#print(doc.text)
	rel_to_tif = {'BEFORE':'TIF-PAST','AFTER':'TIF-FUTURE','INCLUDES':'TIF-ENTIRE','IS_INCLUDED':'TIF-DCT'}
	normalized_values = get_normalized_values(doc)
	timex3_spans = set([sp for lab in doc.get_span_labels_by_regex('value:\d\d\d\d*') for sp in doc.span_annotations[lab]])
	conversion = {sp:lab for lab in normalized_values for sp in doc.span_annotations[lab]}
	if (0,0) in conversion:
		dct = normalized_values[conversion[(0,0)]]
		#print('DCT:', dct, spans[(0,0)])
	

	for timex3_span in timex3_spans:
		value = [v for v in doc.reverse_span_annotations[timex3_span] if v[:5]=='value'][0]
		if not timex3_span == (0,0) and value in normalized_values:
		
			corr_time = normalized_values[value]
			rel = get_temp_rel(corr_time, dct)
			tif = rel_to_tif[rel]
			#print(doc.span_to_string(timex3_span),rel, tif)

			# Getting last token of TIMEX3
			token_indices = doc.span_to_tokens(timex3_span, token_index=True)
			last_token_span = doc.spans[token_indices[-1]]
			
			if not last_token_span in doc.reverse_span_annotations:
				doc.reverse_span_annotations[last_token_span] = []
			doc.reverse_span_annotations[last_token_span].append(tif)
			if not tif in doc.span_annotations:
				doc.span_annotations[tif]=[]
			doc.span_annotations[tif].append(last_token_span)
		
		
	
	
def get_dur_from_value(value):
	
	
	#print('>>>>>>>>>', value)
	
	pattern, res=None, None
	for r in [r'\d\d\d\d-\d\d$', r'P\d+Y$',r'P\d+M$',r'P\d+W$',r'P\d+D$', r'PT(\d+)M(\d+)S$', r'PRESENT_REF$',r'\d\d\d\d-[^d][^d]$', r'\d\d\d\d-\d\d-\d\d$', '\d\d\d\d$','FUTURE_REF$',r'\d\d\d\d-\d\d-\d\dT\d\d$', r'\d\d\d$','PAST_REF$',r'XXXX-Q\d$', 'PXD$','PXY$','PXM$','PXW$', 'PT\d+H$','PT\d+M$']:
		if re.match(r, value):
			pattern, res = r, re.match(r, value)
	if not res:
		return None
	#else:
	#	print('match!',pattern)
	
	unit_to_seconds = {'D':60*60*24, 'Y':365*60*60*24, 'W':7*60*60*24,'Q':91*60*60*24,'M':30*60*60*24, 'H':60*60}	
	
	seconds = None
	if pattern in [r'P\d+Y$',r'P\d+M$',r'P\d+W$',r'P\d+D$']:		
		number, unit = int(value[1:-1]), value[-1]
		seconds = number * unit_to_seconds[unit]
	if pattern == r'\d\d\d\d-\d\d$':
		seconds = unit_to_seconds['M']
	if pattern == r'PT(\d+)M(\d+)S$':
		seconds = int(res.groups()[0]) * 60 + int(res.groups()[1])
	if pattern == r'\d\d\d\d-[^d][^d]$':
		seconds = unit_to_seconds['Q']
	if pattern == r'\d\d\d\d-\d\d-\d\d$':
		seconds = unit_to_seconds['D']
	if pattern == '\d\d\d\d$':
		seconds = unit_to_seconds['Y']
	if pattern == r'\d\d\d\d-\d\d-\d\dT\d\d$':
		seconds = unit_to_seconds['H']
	if pattern == r'\d\d\d$':
		seconds == unit_to_seconds['Y'] * 10
	if pattern == r'XXXX-Q\d$':
		seconds == unit_to_seconds['Q']
	if pattern in ['PXD$','PXY$','PXM$','PXW$']: # several days/years/weeks
		unit = value[-1]
		seconds = unit_to_seconds[unit] * 3
	if pattern == 'PT\d+H$':
		number = int(value[2:-1])
		seconds = unit_to_seconds['H'] * number
	if pattern =='PT\d+M$':
		number = int(value[2:-1])
		seconds = 60 * number
	
	#print(value,'>>', seconds)
		
	return seconds



	
		
def get_num_tlinks(corpus):
	tlinks = {}
	for doc in corpus:
		for label, annotations in doc.span_pair_annotations.items():
			if not label in tlinks:
				tlinks[label]=0
			tlinks[label] += len(annotations)
	return tlinks			

def test():

	testfolder="/home/tuur/Desktop/TACL/data/TempEval3/train/TBAQ-cleaned/TimeBank/"
	documents = read_timebank_folder(testfolder)
	reimers_annotations= '/home/tuur/Desktop/TACL/data/Reimersetal2016/event-times.tab'
	documents = extend_annotations_reimers(documents, reimers_annotations)
	timebank_dense = '/home/tuur/Desktop/TACL/data/TimebankDense/TimebankDense.T3.txt'
	documents = extend_annotations_timebankdense(documents, timebank_dense, replace=True)
	brat_out = '/home/tuur/Desktop/SpaRad/ENVIRONMENT/data/temporal/reimers/'
	for doc in documents:
		doc.write_to_brat(brat_out, span_label_filter='EType:|Reimersetal:|eid:|tid:|value:|class:|type:', span_pair_label_filter='.*')
