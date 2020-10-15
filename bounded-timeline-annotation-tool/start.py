import eel, glob, os, argparse, bs4, re
import plotly.offline as py
import plotly.io as pio
from numpy import recarray
import random
import datetime
from lib.EventTimeAnnotation import *
from lxml import etree as ET
from copy import copy
from lib.i2b2 import write_annotation_to_event_in_doc
import plotly.figure_factory as ff

#eel, bs4, plotly, numpy




eel.init('web')
eel.start('main.html', block=False)

def get_data_directory_structure(top_dir='data'):
    print(top_dir)
    absolute_top_dir = os.getcwd() + '/' + top_dir
    dir_to_files = {}
    for f in glob.iglob(absolute_top_dir + '/**/*.xml', recursive=True):
        f = f.replace(absolute_top_dir,'')
        dir = '/'.join(f.split('/')[:-1])
        file = f.split('/')[-1]
        dir_to_files[dir] = [file] if not dir in dir_to_files else dir_to_files[dir] + [file]

    for dir in dir_to_files:
        try:
            dir_to_files[dir] = sorted(dir_to_files[dir],key=lambda x: int(x.split('_')[-1][:-4]))
        except:
            dir_to_files[dir] = sorted(dir_to_files[dir])


#        print(dir_to_files[dir])

        # # ------------------------------->
        # random.seed(1)
        # random.shuffle(dir_to_files[dir]) # randomly shuffle the directories files into a list
        # if len(dir_to_files[dir]) > 10:
        #     print(dir)
        #     fraction = 3
        #     split_index = int(float(len(dir_to_files[dir])) / fraction) # get a fraction that should be shared
        #     close = dir_to_files[dir][:split_index] # close documents ensure annotated annotated always a certain proportion of the same documents while annotating so agreement can be calculated
        #     print('shared',close)
        #     print('shared fraction: ',fraction)
        #     far = dir_to_files[dir][split_index:] # far documents are positioned on very different positions for both annotators to ensure all data will be annotated by at least one annotator as quickly as possible
        #     random.seed(hash(dir))
        #     random.shuffle(far)
        #     dir_to_files[dir]=[]
        #     print('>>>>',len(close) + len(far))
        #     for i in range(len(close)):
        #         dir_to_files[dir] += [close[i]]
        #         dir_to_files[dir] += far[max((i-1)*fraction, 0):i*fraction]
        #     print(len(dir_to_files[dir]))




    return dir_to_files

@eel.expose
def set_dir(dir):
    global current_directory
    current_directory = dir
    print('setting current dir', dir)

@eel.expose
def set_file(file_id):
    global current_doc_id, current_event
    file_id = int(file_id)
    current_doc_id = file_id
    current_event = None
    print('setting current doc', data_directory_structure[current_directory][current_doc_id])

@eel.expose
def get_data_directories():
    global data_directory_structure
    return data_directory_structure



@eel.expose
def pprint(s):
    print(s)

@eel.expose
def set_propagation():
    global propagation_mode
    if propagation_mode:
        propagation_mode = False
    else:
        propagation_mode = True

@eel.expose
def load_current_document():
    global data_directory_structure, current_directory, current_doc_id, annotated_events
    annotated_events = {}
    if len(data_directory_structure) == 0:
        print('no data found...')
        eel.setDocument("", "no data found...")
    else:
        doc_path =  get_current_doc_path()
        text = parse_i2b2_xml(doc_path)
        title = "Document " + doc_path.split('/')[-1] + " (" + str(current_doc_id + 1) + '/' + str(len(data_directory_structure[current_directory])) +")"
        eel.setDocument(text, title)
        eel.colorEvent(current_event, "#24abe7");
        eel.boldEvent(current_event);

        if current_event:
            eel.setEventText(current_event)
            eel.showAnnotationInputs()
        else:
            eel.hideAnnotationInputs()
        print('loading doc', doc_path)
    set_timeline()

def get_current_doc_path():
    global data_directory_structure, current_directory, current_doc_id
    file_name = data_directory_structure[current_directory][current_doc_id % len(data_directory_structure[current_directory])]
    doc_path = 'data/' + current_directory + '/' + file_name
    return doc_path


def parse_i2b2_xml(path):
    with open(path) as f:
        xml = f.read()
        html_txt = get_text_with_event_tags(xml)
        return html_txt

def get_text_with_event_tags(i2b2_xml):
    #print(i2b2_xml)
    global current_event, annotated_events
    soup = bs4.BeautifulSoup(i2b2_xml, features="xml")
    text = soup.find('TEXT').get_text()
    events = soup.find_all('EVENT')
    splitted_text = []
    indices = []
    span_to_event = {}
    annotated = set()
    for event in events:
        (s,e) = int(event['start']), int(event['end'])
        span_to_event[(s,e)] = event
    html_tagged_text = ""
    sorted_spans = sorted(span_to_event, key=lambda x: x[0])
    prev_end = 0
    for span in sorted_spans:
        html_tagged_text += text[prev_end:span[0]]
        event = span_to_event[span]

        event_has_annotations = event.has_attr('temporally-annotated') and event['temporally-annotated'] == "YES"

        classes = "event"
        if event_has_annotations:

            annotated_events[event['id']] = event
            annotation = get_annotations_from_event_xml(event)
            if not annotation.is_complete():
                classes += " todo"
            else:

                if not annotation.is_consistent()[0]:
                    classes += " inconsistent"

                elif annotation.has_exact_boundaries():
                    classes += " exact"
                elif annotation.has_close_duration_boundaries():
                    classes += " close"
                else:
                   classes += " done"
        else:
            classes += " todo"

        classes += " " + event['type']
        #classes = "event done " + event['type'] if event_is_annotated and event_consistent else "event inconsistent done " + event['type'] if event_is_annotated and not event_consistent else 'event todo ' + event['type']
        html_tagged_text += '<span class="' + classes + '"id=' + event['id']+ ' onclick="eel.selectEvent(`'+event['id']+'`);">'
        html_tagged_text += text[span[0]:span[1]]
        html_tagged_text += "<span class='tooltiptext'>"+event['id']+"</span>"
        html_tagged_text += '</span>'
        prev_end = span[1]
    html_tagged_text += text[prev_end:]
    html_tagged_text = html_tagged_text.replace('\n','<br/>')
    return html_tagged_text

def set_timeline():
    global annotated_events, current_event
    doc_path = get_current_doc_path()
    df = []
    event_text_start_and_ends = {}
    for e_id, event in annotated_events.items():
        # print(e_id, type(event), event)
        event_annotations = get_annotations_from_event_xml(event)
        # print(e_id, event_annotations)
        if not event_annotations.is_complete():
            continue

        category = 'Selected' if e_id == current_event else 'Event'
        event_timeline_dict = dict(Task=event['text'], datetimeStart=event_annotations.get_most_likely_start(), Start=str(event_annotations.get_most_likely_start()), Finish=str(event_annotations.get_most_likely_end()), Resource=category)
        #print(event_timeline_dict)
        df.append(event_timeline_dict)

    df = sorted(df, key=lambda x:x['datetimeStart'], reverse=True)

    colors = dict(Selected='rgb(82,189,236)',
                  Event='rgb(17, 110, 138)')

    dimensions = eel.getTimelineDimensions()()
    eel.sleep(0.001)
    timeline_width = int(dimensions[0] * 0.45)
    timeline_height = int(dimensions[1])


    if len(df) > 0:
        fig = ff.create_gantt(df, colors=colors, index_col='Resource', title='Most Likely Timeline',
                              show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True, width=timeline_width,
                              height=timeline_height)

        #print(fig['layout'])
        #fig['layout']

        for i, datapoint in list(enumerate(fig['data']))[:len(df)]:
            if df[i]['Resource'] == "Selected":

                def set_timeline():
                    global annotated_events, current_event
                    doc_path = get_current_doc_path()
                    df = []
                    event_text_start_and_ends = {}
                    for e_id, event in annotated_events.items():
                        # print(e_id, type(event), event)
                        event_annotations = get_annotations_from_event_xml(event)
                        # print(e_id, event_annotations)

                        # print(event)
                datapoint['marker']['color'] = "rgb(116,202,239)"
                fig['layout']['yaxis']['ticktext'][i] = fig['layout']['yaxis']['ticktext'][i]+" >"
            else:
                datapoint['marker']['color'] = "rgb(17, 110, 138)"
            fig['layout']['yaxis']['automargin']=True

        py.offline.plot(fig, filename='web/timeline.html', auto_open=False)
        eel.refreshTimeLine('timeline.html')
    else:
        eel.refreshTimeLine('blank.html')

@eel.expose
def remove_selected_annotations_from_current_doc():
    global current_event
    temporal_annotation = EventTimeAnnotation(current_event)

    doc_path = get_current_doc_path()

    with open(doc_path, 'r') as f:
        i2b2_xml = f.read()
        soup = bs4.BeautifulSoup(i2b2_xml, features="xml")

        # find the event with the annotations id
        event = soup.find("EVENT",attrs={'id':temporal_annotation.event_id})

        # remove existing temporal annotations
        for attr in ['most-likely-duration','lowerbound-duration','upperbound-duration','most-likely-start','lowerbound-start','upperbound-start','most-likely-end','lowerbound-end','upperbound-end','temporally-annotated']:
            if event.has_attr(attr):
                del event[attr]
    # write updated xml back to original file
    with open(doc_path, 'w') as f:
        f.write(str(soup))
    eel.colorSaveButton('#ff6600')




@eel.expose
def selectEvent(id):
    global current_event, copy_mode

    if copy_mode:
        remember_id = current_event

    current_event = id
    print('selected',current_event)


    annotations = get_annotations_from_event_xml(annotated_events[current_event]) if current_event in annotated_events else False
    print('annotations:',annotations)
    if annotations:
        print('annotated:',annotations.annotated)

    if not annotations:
        eel.openFreshAnnotations()
        print('no annotations founds')
        if not copy_mode:
            eel.colorSaveButton('#ff6600')
        load_current_document()
    elif not copy_mode:
        eel.openFreshAnnotations()
        print('loading annotations')
        for field in annotations.annotated:
            setAnnotationField(annotations, field)
        eel.colorSaveButton('#116E8A')
        load_current_document()
    elif copy_mode:
        if copy_mode in ['duration','start','end']:
            setAnnotationField(annotations, copy_mode)
            eel.refreshAnnotationOptions()
        elif copy_mode =='full':
            eel.openFreshAnnotations()
            for field in annotations.annotated:
                setAnnotationField(annotations, field)
        elif copy_mode =='start_at_end':
            tmp = EventTimeAnnotation('NOID') # create temporary annotation, with as the start the end of the selected annotation
            tmp.add_start(annotations.e, annotations.e_lower, annotations.e_upper)
            print(annotations.s, annotations.d, annotations.e)
            print(tmp.s)
            setAnnotationField(tmp, 'start')
        elif copy_mode =='end_at_start':
            tmp = EventTimeAnnotation('NOID') # create temporary annotation, with as the end the start of the selected annotation
            tmp.add_end(annotations.s, annotations.s_lower, annotations.s_upper)
            setAnnotationField(tmp, 'end')

        current_event = remember_id
        resetCopyMode()



def setAnnotationField(annotations, field):
    if field =='duration':
        eel.setDuration(annotations.d.as_list_of_strings() + annotations.d_lower.as_list_of_strings() + annotations.d_upper.as_list_of_strings())
    elif field=='start':
        eel.setStart([str(annotations.s).replace(' ', 'T'), str(annotations.s_lower).replace(' ', 'T'), str(annotations.s_upper).replace(' ', 'T')])
    elif field=='end':
        eel.setEnd([str(annotations.e).replace(' ', 'T'), str(annotations.e_lower).replace(' ', 'T'), str(annotations.e_upper).replace(' ', 'T')])
    else:
        print('error: unknown annotation field:',field)

@eel.expose
def setCopyMode(mode):
    global copy_mode
    copy_mode = mode
    eel.setCopyVisuals(copy_mode)
    print('copymode:',mode)

@eel.expose
def resetCopyMode():
    global copy_mode
    copy_mode = False
    eel.resetCopyVisuals()


@eel.expose
def search_string_in_all_documents(search_string):

    hits = {}
    for directory in data_directory_structure:

        for doc in data_directory_structure[directory]:
            path = 'data/' + directory +'/' + doc
            ppath = directory + '/' + doc

            with open(path) as f:
                xml = f.read()
                soup = bs4.BeautifulSoup(xml, features="xml")
                text = soup.find('TEXT').get_text()
                count = text.count(search_string)
                if count > 0:
                    hits[ppath] = count
    alert = "Found " + str(len(hits)) + " hits!\n"
    for dpath,c in hits.items():
        line = "\n" + str(c) + " x in " + dpath
        alert += line
    eel.popup(alert)



@eel.expose
def save_annotations(values):
    global current_event
    print(values)

    annotation = EventTimeAnnotation(current_event)

    # Adding the duration

    if 'duration-Y' in values:
        d_string = ''.join([values['duration-'+unit]+unit for unit in ['Y','M','D','H','m']])
        d = CalenderDuration.read_from_string(d_string)

        dmin_string = ''.join([values['min-duration-' + unit] + unit for unit in ['Y', 'M', 'D', 'H', 'm']])
        dmin = CalenderDuration.read_from_string(dmin_string)

        dmax_string = ''.join([values['max-duration-' + unit] + unit for unit in ['Y', 'M', 'D', 'H', 'm']])
        dmax = CalenderDuration.read_from_string(dmax_string)

        if not (dmin.in_minutes() <= d.in_minutes() and d.in_minutes() <= dmax.in_minutes()):
            eel.popup("Annotation mistake: the most likely duration does not lie between the minimum and maximum durations! Annotations not saved!")
            return False
        else:
            annotation.add_duration(d, dmin, dmax)

    start_or_end = False
    if 'start-time' in values:
        incorrectly_filled = [v for v in ['start-time','start-min', 'start-max'] if values[v]==""]
        if len(incorrectly_filled) > 0:
            eel.popup("Annotation mistake: start-time is not annotated completely! Annotations not saved!")
            return False
        else:
            print([values[v].replace('T',' ') for v in ['start-time','start-min', 'start-max']])
            s, smin, smax = [CalenderPoint.read_from_string(values[v].replace('T',' ')) for v in ['start-time','start-min', 'start-max']]
            annotation.add_start(s, smin, smax)
            start_or_end = "start"

    if 'end-time' in values:
        incorrectly_filled = [v for v in ['end-time','end-min', 'end-max'] if values[v]==""]
        if len(incorrectly_filled) > 0:
            eel.popup("Annotation mistake: end-time is not annotated completely! Annotations not saved!")
            return False
        else:
            print([values[v].replace('T',' ') for v in ['end-time','end-min', 'end-max']])
            e, emin, emax = [CalenderPoint.read_from_string(values[v].replace('T',' ')) for v in ['end-time','end-min', 'end-max']]
            annotation.add_end(e, emin, emax)
            start_or_end = "end"

    if not start_or_end:
        eel.popup("Annotation mistake: No start or end annotated! Annotations not saved!")
        return False
    else:

        if propagation_mode:
            print('======= BOUND PROPAGATION MODE ======= >>>')
            print(current_event)
            print(get_current_doc_path())

            # 1) get the initially annotated annotations (old ones) if there are any
            with open(get_current_doc_path(),'r') as f:
                soup = bs4.BeautifulSoup(f.read(), features="xml")
                event_xml = soup.find('EVENT',attrs={'id':str(current_event)})

            if event_xml:
                old_annotation = get_annotations_from_event_xml(event_xml)
                # 2) check what start/end points changed --> duration --> normal
                start_changed = 'start-time' in values and (old_annotation.s.point != annotation.s.point or old_annotation.s_lower.point != annotation.s_lower.point or old_annotation.s_upper.point != annotation.s_upper.point)
                end_changed = 'end-time' in values and (old_annotation.e.point != annotation.e.point or old_annotation.e_lower.point != annotation.e_lower.point or old_annotation.e_upper.point != annotation.s_upper.point)
                print('NEW S:', start_changed)
                print('NEW E:', end_changed)
                # 2.1) if it is a start or end: --if no --> normal

                # 3) find all other event annotation with such a point (start or end) --> if no --> normal

                all_annotated_events = soup.find_all('EVENT',attrs={'temporally-annotated':"YES"})
                print('Num annotated events:',len(all_annotated_events))

                updated = []
                for prop_event in all_annotated_events: # for each event to which the points could be propagated (prop_event)
                    if not prop_event['id'] == str(current_event):
                        prop_event_ann = get_annotations_from_event_xml(prop_event)


                        if not prop_event_ann.is_complete():
                            continue
                        num_modifications = 0
                        print('checking propagation:',prop_event.id)

                        if prop_event_ann.has_the_same_values(old_annotation): # if two events have the exact same start AND end just propagate the full annot updatedations
                            print('! same values:',prop_event_ann.event_id )
                            new_prop_event_ann = copy(annotation)
                            new_prop_event_ann.event_id = prop_event_ann.event_id
                            num_modifications = 2
                        else:
                            new_prop_event_ann = EventTimeAnnotation(prop_event_ann.event_id)

                            if 'start' in prop_event_ann.annotated:
                                # if the start of the annotated event equals the start of another event
                                if prop_event_ann.has_the_same_start(old_annotation): # if they just share the start: only propagate the start
                                    new_prop_event_ann.add_start(annotation.s, annotation.s_lower, annotation.s_upper)
                                    num_modifications += 1
                                # if start of the annotated event equals the end of another event
                                if old_annotation.e.point == prop_event_ann.s.point and old_annotation.e_lower.point == prop_event_ann.s_lower.point and old_annotation.e_upper.point == prop_event_ann.s_upper.point:
                                    new_prop_event_ann.add_end(annotation.s, annotation.s_lower, annotation.s_upper)
                                    num_modifications += 1

                            if 'end' in prop_event_ann.annotated:
                                if prop_event_ann.has_the_same_end(old_annotation):
                                    new_prop_event_ann.add_end(annotation.e, annotation.e_lower, annotation.e_upper)
                                    num_modifications += 1

                                if old_annotation.e.point == prop_event_ann.s.point and old_annotation.e_lower.point == prop_event_ann.s_lower.point and old_annotation.e_upper.point == prop_event_ann.s_upper.point:
                                    new_prop_event_ann.add_start(annotation.e, annotation.e_lower, annotation.e_upper)
                                    num_modifications += 1



                            if 'duration' in prop_event_ann.annotated and num_modifications == 1:
                                new_prop_event_ann.add_duration(prop_event_ann.d, prop_event_ann.d_lower, prop_event_ann.d_upper)
                                num_modifications +=1

                            if 'start' in prop_event_ann.annotated and num_modifications == 1:
                                new_prop_event_ann.add_start(prop_event_ann.s, prop_event_ann.s_lower, prop_event_ann.s_upper)
                                num_modifications += 1

                        if new_prop_event_ann.is_complete():
                            write_annotation_to_event_in_doc(get_current_doc_path(), new_prop_event_ann)
                            updated.append(new_prop_event_ann.event_id)
                        elif num_modifications == 0:
                                pass
                        else:
                            print('!?!?!?!',num_modifications,new_prop_event_ann.annotated)

                if len(updated) > 0:
                    eel.popup("Updated the following additional events" + str(updated)) # TODO: yes / no option instead of just OK

                    # 4) adapt those event annotations
                    # 5) ask in a popup to change other annotation as well, write them all to file

            print('======= END BOUND PROPAGATION MODE ======= <<<')

            write_annotation_to_event_in_doc(get_current_doc_path(), annotation)
        else:
            write_annotation_to_event_in_doc(get_current_doc_path(), annotation)

        eel.colorSaveButton('#116E8A');

    load_current_document()







    # make annotation object
    # save to file




data_directory_structure = get_data_directory_structure('data')
current_directory = list(data_directory_structure.keys())[0]
current_doc_id = 0
annotated_events = {}
copy_mode = False
current_event = None
propagation_mode = False
print('current_directory:',current_directory)
print('current document:',data_directory_structure[current_directory][current_doc_id])
load_current_document()

while True:

    eel.sleep(1);
    print('.', current_event, datetime.datetime.now())


    #if current_event:
     #   tempann = EventTimeAnnotation(current_event)
      #  #tempann.add_start()
       # add_annotation_to_event_in_doc(get_current_doc_path(), tempann)