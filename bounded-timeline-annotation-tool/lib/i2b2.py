import glob, codecs, re, os, shutil
from bs4 import BeautifulSoup
from lib.EventTimeAnnotation import EventTimeAnnotation, CalenderDuration, CalenderPoint
from lib.data import Text
import bs4


def write_annotation_to_event_in_doc(doc_path, temporal_annotation):

    # read xml
    with open(doc_path, 'r') as f:
        i2b2_xml = f.read()
        soup = bs4.BeautifulSoup(i2b2_xml, features="xml")

        # find the event with the annotations id
        event = soup.find("EVENT",attrs={'id':temporal_annotation.event_id})

        # remove existing temporal annotations
        for attr in ['most-likely-duration','lowerbound-duration','upperbound-duration','most-likely-start','lowerbound-start','upperbound-start','most-likely-end','lowerbound-end','upperbound-end']:
            if event.has_attr(attr):
                del event[attr]

        # add duration attributes
        if 'duration' in temporal_annotation.annotated:
            event['most-likely-duration'] = str(temporal_annotation.d)
            event['lowerbound-duration'] = str(temporal_annotation.d_lower)
            event['upperbound-duration'] = str(temporal_annotation.d_upper)

        # add start or end attributes
        if 'start' in temporal_annotation.annotated:
            event['most-likely-start'] = str(temporal_annotation.s)
            event['lowerbound-start'] = str(temporal_annotation.s_lower)
            event['upperbound-start'] = str(temporal_annotation.s_upper)

        if 'end' in temporal_annotation.annotated:
            event['most-likely-end'] = str(temporal_annotation.e)
            event['lowerbound-end'] = str(temporal_annotation.e_lower)
            event['upperbound-end'] = str(temporal_annotation.e_upper)

        event['temporally-annotated'] = "YES"

    # write updated xml back to original file
    with open(doc_path, 'w') as f:
        f.write(str(soup))


def read_i2b2_folder(folder, verbose=1, conflate_digits=False, pos=False, lowercase=True):
    docs = []
    total_tlinks = 0
    for file_path in glob.glob(folder + "*.xml"):
        with codecs.open(file_path, 'r') as f:
            num_events, num_timex3, num_tlinks = 0,0,0
            xml_str = f.read()
            xmlSoup = BeautifulSoup(xml_str, 'xml')
            raw_txt =  xmlSoup.find_all('TEXT')[0].contents[0]

            entity_labels = ['EType:EVENT', 'EType:TIMEX3']
            label_to_spans = {l: [] for l in entity_labels}
            label_to_span_pairs = {}
            doc_id = file_path.split('/')[-1]


            for event_tag in xmlSoup.find_all('EVENT'):
                num_events += 1
                text_span = (int(event_tag.attrs['start']), int(event_tag.attrs['end']))
                for attr, value in event_tag.attrs.items():
                    label = attr +":"+value
                    if not label in label_to_spans:
                        label_to_spans[label]= []
                    label_to_spans[label].append(text_span)
                label_to_spans['EType:EVENT'].append(text_span)


            for timex_tag in xmlSoup.find_all('TIMEX3'):
                num_timex3 += 1
                text_span = (int(timex_tag.attrs['start']), int(timex_tag.attrs['end']))
                for attr, value in timex_tag.attrs.items():
                    label = attr +":"+value
                    if not label in label_to_spans:
                        label_to_spans[label]= []
                    label_to_spans[label].append(text_span)
                label_to_spans['EType:TIMEX3'].append(text_span)

            for timex_tag in xmlSoup.find_all('SECTIME'):
                num_timex3 += 1
                text_span = (int(timex_tag.attrs['start']), int(timex_tag.attrs['end']))
                for attr, value in timex_tag.attrs.items():
                    label = attr +":"+value
                    if not label in label_to_spans:
                        label_to_spans[label]= []
                    label_to_spans[label].append(text_span)
                label_to_spans['EType:TIMEX3'].append(text_span)



            for tlink_tag in xmlSoup.find_all('TLINK'):
                eid1, eid2 = 'id:'+tlink_tag.attrs['fromID'], 'id:'+tlink_tag.attrs['toID']
                if not eid1 in label_to_spans or not eid2 in label_to_spans:
                    continue
                sp1, sp2 = label_to_spans[eid1][0] , label_to_spans[eid2][0]
                num_tlinks += 1
                for attr, value in tlink_tag.attrs.items():
                    label = attr +":"+value
                    if not label in label_to_spans:
                        label_to_spans[label]= []
                label_to_spans[label].append((sp1,sp2))

            text = Text(raw_txt, span_annotations=label_to_spans, span_pair_annotations=label_to_span_pairs, id=doc_id,conflate_digits=conflate_digits, pos=pos, lowercase=lowercase, file_path=file_path)
            docs.append(text)
            if verbose:
                print(doc_id, '\tevents:', num_events, 'timex3:',num_timex3, 'tlinks:', num_tlinks)
        total_tlinks += num_tlinks
    return docs




"""
# read TimeAnno annotation from xml
timeanno_annotation = EventTimeAnnotation(event_id=eid, event_string=event_text)
if 'most-likely-start' in event_tag.attrs:
    print(event_tag.attrs['most-likely-start'])
    s_likely, s_lower, s_upper = CalenderPoint.read_from_string(
        event_tag.attrs['most-likely-start']), CalenderPoint.read_from_string(
        event_tag.attrs['lowerbound-start']), CalenderPoint.read_from_string(event_tag.attrs['upperbound-start'])
    timeanno_annotation.add_start(s_likely, s_lower, s_upper)
if 'most-likely-end' in event_tag.attrs:
    print(event_tag.attrs['most-likely-end'])
    e_likely, e_lower, e_upper = CalenderPoint.read_from_string(
        event_tag.attrs['most-likely-end']), CalenderPoint.read_from_string(
        event_tag.attrs['lowerbound-end']), CalenderPoint.read_from_string(event_tag.attrs['upperbound-end'])
    timeanno_annotation.add_end(e_likely, e_lower, e_upper)
if 'most-likely-duration' in event_tag.attrs:
    d_likely, d_lower, d_upper = CalenderDuration.read_from_string(
        event_tag.attrs['most-likely-duration']), CalenderDuration.read_from_string(
        event_tag.attrs['lowerbound-duration']), CalenderDuration.read_from_string(
        event_tag.attrs['upperbound-duration'])
    timeanno_annotation.add_duration(d_likely, d_lower, d_upper)
timeanno_annotation.infer_complete_annotation()
"""