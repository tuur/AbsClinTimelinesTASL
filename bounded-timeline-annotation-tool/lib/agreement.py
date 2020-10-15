import bs4, os, shutil
from lib.EventTimeAnnotation import get_annotations_from_event_xml, CalenderDuration
import numpy as np
import plotly.figure_factory as ff
import plotly.offline as py
import plotly.io as pio


def get_all_annotations_from_document(doc, event_types=None):
    annotations = []
    event_strings = []
    with open(doc.file_path, 'r') as f:
        soup = bs4.BeautifulSoup(f.read(), features="xml")
        for event in soup.find_all('EVENT'):
            if not event_types or event['type'] in event_types:
                ann = get_annotations_from_event_xml(event)
                event_strings.append(event['text'])
                if len(ann.annotated) > 1:
                   annotations.append(ann)
    return  annotations, event_strings

class AgreementSummary:

    def __init__(self, aligned_document_pairs, event_types=None, per_document=False, latex=False):
        self.aligned_document_pairs=aligned_document_pairs
        self.viz_timeline_differences()

        experiments = [None]
        if event_types:
            experiments = [[e] for e in event_types] + [list(event_types)]
            print('EXPERIMENTS:',experiments)
        agreement_per_exp = {}
        for exp in experiments:
            all_a1_annotations = []
            all_a2_annotations = []
            print("--------",exp,"--------")
            for d1, d2 in aligned_document_pairs:
                a1_annotations,_ = get_all_annotations_from_document(d1, exp)
                a2_annotations,_ = get_all_annotations_from_document(d2, exp)

                if per_document:
                    print('\n======== DOC', d1.id, '========')
                    Agreement(a1_annotations, a2_annotations)
                else:
                    for a1a in a1_annotations:
                        a1a.event_id = d1.id +'-'+a1a.event_id
                        all_a1_annotations.append(a1a)
                    for a2a in a2_annotations:
                        a2a.event_id = d1.id +'-'+a2a.event_id
                        all_a2_annotations.append(a2a)

            if not per_document:
                agr = Agreement(all_a1_annotations, all_a2_annotations)

                name = exp[0] if len(exp)==1 else 'All'

                agreement_per_exp[name] = agr

        if latex:
            print('+++ LATEX +++')
            print(self.make_latex(agreement_per_exp))

    def viz_timeline_differences(self, out_dir="./timeline_comparisons"):
        if os.path.exists(out_dir):
	        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for d1, d2 in self.aligned_document_pairs:
            a1_annotations,a1_event_strings = get_all_annotations_from_document(d1)
            a2_annotations,a2_event_strings = get_all_annotations_from_document(d2)
            path = out_dir + "/" + d1.id.split('_')[-1][:-4] + '.html'
            self.make_timeline(a1_annotations, a2_annotations, a1_event_strings, a2_event_strings, path)



    def make_timeline(self, a1_anns, a2_anns, a1_strings, a2_strings, path):
        df = []

        event_annotations = list(zip(a1_anns,a2_anns, a1_strings, a2_strings))
        event_annotations = sorted(event_annotations, key = lambda x: x[0].s.point, reverse=True) # show timeline in order of a1's event annotations
        colors = dict(A1='rgb(51,51,255)',
                      A2='rgb(255, 128, 0)')
        for a1, a2, es1, es2 in event_annotations:
            # A1 annotation

            s1 = a1.s_lower.point
            e1 = a1.e_upper.point
            event_timeline_dict_a1 = dict(Task=es1, datetimeStart=s1,
                                       Start=str(s1),
                                       Finish=str(e1), Resource="A1")

            s2 = a2.s_lower.point
            e2 = a2.e_upper.point
            event_timeline_dict_a2 = dict(Task=es2, datetimeStart=s2,
                                       Start=str(s2),
                                       Finish=str(e2), Resource="A2")
            df.append(event_timeline_dict_a1)
            df.append(event_timeline_dict_a2)

            fig = ff.create_gantt(df, colors=colors, index_col='Resource', title='Most Likely Timeline',
                                  show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True, width=2000, height=1000)

        py.offline.plot(fig, filename=path, auto_open=False)
        print('written timeline comparison to', path)




    def make_latex(self, agreement_per_exp_dict):
        num_categories = len(agreement_per_exp_dict)
        catsa = agreement_per_exp_dict.items()
        tex = "\\begin{tabular}{l" + num_categories*"c" + "}\hline\\n"
        for cat,a in catsa:
            name = cat[0] + cat[1].lower() +'.' if not cat == 'All' else cat
            tex += "&" + name
        tex += "\\\\\hline\nEvents"
        for _,a in catsa: # Number of Events
            tex += "&" + str(len(a.A1s))
        tex += "\\\\\n$\Tilde{u}(x)$"
        for _,a in catsa: # Overlap Alpha Start
            tex += "&" + str(round(a.u/60, 0))[:-2]+'h'

        tex +='\\\\Bounds:'
        tex += "\\\\\n\\hspace{.15cm}$\\alpha_{b}$ $x_s^*$"
        for _,a in catsa: # Overlap Alpha Start
            tex += "&" + str(round(a.alpha_ob['start'], 2))[1:]
        tex += "\\\\\n\\hspace{.15cm}$\\alpha_{b}$ $x_e^*$"
        for _,a in catsa:# Overlap Alpha End
            tex += "&" + str(round(a.alpha_ob['end'], 2))[1:]
        tex += "\\\\\n\\hspace{.15cm}$\\alpha_{b}$ $x_d^*$"
        for _,a in catsa:# Overlap Alpha Duration
            tex += "&" + str(round(a.alpha_ob['duration'], 2))[1:]
        #tex += "\\\\\n$\kappa_{om}$ $\\hat{x}_d$"
        #for _,a in catsa:# Order Magnitude Kappa Duration
        #    tex += "&" + str(round(a.kappa_om['duration'], 2))[1:]

        tex +='\\\\Intervals:'
        #tex += "\\\\\n\\hspace{.15cm}$\kappa_{io}$"
        #for _,a in catsa: # Interval Overlap Agreement
        #    tex += "&" + str(round(a.kappa_io, 2))[1:]
        tex += "\\\\\n\\hspace{.15cm}$\\alpha_{c}$"
        for _,a in catsa: # Interval Overlap Agreement
            tex += "&" + str(round(a.kappa_ic, 2))[1:]
        tex += "\\\\\n\\hspace{.15cm}$\\alpha_{p}$"
        for _,a in catsa: # Interval Overlap Agreement
            tex += "&" + str(round(a.kappa_p, 2))[1:]
        tex +='\\\\Order:'

        tex += "\\\\\n\\hspace{.15cm}$\kappa_{to}$ $\\hat{x}_s$"
        for _,a in catsa:# Temporal Order Kappa Start
            tex += "&" + str(round(a.kappa_to['start'], 2))[1:]
        tex += "\\\\\n\\hspace{.15cm}$\kappa_{to}$ $\\hat{x}_e$"
        for _,a in catsa:# Temporal Order Kappa Start
            tex += "&" + str(round(a.kappa_to['end'], 2))[1:]


        tex += "\\\\\hline\n\end{tabular}"

        return tex


class Agreement:

    def __init__(self, A1s, A2s, certainty_filter = 1.0, multiply_uncertainty=1):

        self.A1s = {a1.event_id: a1 for a1 in A1s}
        self.A2s = {a2.event_id: a2 for a2 in A2s}

        self.shared_ids = set([id for id in self.A1s.keys() if id in self.A2s.keys()])

        if multiply_uncertainty:
            self.A1s, self.A2s = {a1.event_id: a1.mult_u(multiply_uncertainty) for a1 in A1s}, {a2.event_id: a2.mult_u(multiply_uncertainty) for a2 in A2s}


        if certainty_filter < 1.0:
            self.shared_ids = self.apply_certainty_filter(certainty_filter)


        self.different_ids = set([id for id in list(self.A1s.keys()) + list(self.A2s.keys()) if not id in self.shared_ids])
        self.kappa_to = {}
        self.kappa_om = {}
        self.alpha_ob = {}
        self.kappa_io = {}
        self.kappa_ic, self.kappa_p = self.kappa_interval_consistency()
        self.u = self.get_uncertainty()
        disagreement_func = lambda x, y: x != y

        print("\n ======= UNCERTAINTY =======")
        print('Median uncertainty', round(self.u,0), '(hours:',round(self.u/60,0),')')
        d = CalenderDuration()
        print('Order of', d.order_of_magnitude(mins=self.u))

        print("\n======== ORDER IAA ========")
        print(len(self.shared_ids), len(self.different_ids))
        for component, component_grapping_function in [('start', lambda x:x.s.point), ('end', lambda x:x.e.point), ('duration', lambda x:x.d.in_minutes())]:

            confusion_matrix, events_involved_in_agreement, events_involved_in_disagreement = self.get_order_confusion_matrix(component_grapping_function)
            kappa, Po, Pe = self.cohens_kappa_from_confusion_matrix_dict(confusion_matrix, disagreement_func)
            self.kappa_to[component] = kappa
            print(component, 'kappa',round(kappa,4), 'Po',round(Po,4), 'Pe',round(Pe,4))
            print(confusion_matrix)

            u_agreement = np.mean([self.A1s[eid].u()+self.A2s[eid].u() for eid in events_involved_in_agreement])
            u_disagreement = np.mean([self.A1s[eid].u()+self.A2s[eid].u() for eid in events_involved_in_disagreement])


            print('agreed:', len(events_involved_in_agreement), 'disagreed:',len(events_involved_in_disagreement))
            # TODO: calc average U_d for each set and print (and print ratio)
            print('uncertainty agreed events:', u_agreement)
            print('uncertainty disagreed events:', u_disagreement)


        print("\n======= Interval Overlap Agreement =======")
        Po_overlap = self.Po_interval_overlap()
        Pe_overlap = self.Pe_interval_overlap()
        self.kappa_io = (Po_overlap - Pe_overlap) / (1 - Pe_overlap)
        print('kappa',round(self.kappa_io, 4), 'Po', round(Po_overlap,4),'Pe', round(Pe_overlap,4))


        print("\n======== ANY OVERLAP ABSOLUTE IAA ========")

        for component, component_grapping_function in [('start', lambda x:(x.s_lower.in_minutes_from_Christ(), x.s_upper.in_minutes_from_Christ())), ('end', lambda x:(x.e_lower.in_minutes_from_Christ(), x.e_upper.in_minutes_from_Christ())), ('duration', lambda x:(x.d_lower.in_minutes(), x.d_upper.in_minutes()))]:
            alpha, Po, Pe = self.partial_agreement_krippendorf_alpha(component_grapping_function)
            self.alpha_ob[component] = alpha
            print(component,'alpha',round(alpha, 4), 'Po',round(Po,4), 'Pe',round(Pe,4))

        print("\n======== DURATION ORDER OF MAGNITUDE ========")
        confusion_matrix = self.get_duration_order_of_magnitude_confusion_matrix()
        kappa, Po, Pe = self.cohens_kappa_from_confusion_matrix_dict(confusion_matrix)
        self.kappa_om['duration']=kappa
        print('kappa',kappa, 'Po',round(Po,4), 'Pe',round(Pe,4))
        print(confusion_matrix)


    def get_uncertainty(self):
        us = []

        for eid in self.shared_ids:
            a1 = self.A1s[eid]
            a2 = self.A2s[eid]
            avg_u = np.mean([a1.u(), a2.u()])
            us.append(avg_u)

        #us = sorted(us)
        return np.median(us)

    def apply_certainty_filter(self, certainty):
        u = []
        for eid in self.shared_ids:
            ua1 = self.A1s[eid].u()
            ua2 = self.A2s[eid].u()
            u.append((eid, ua1 + ua2))
        u = sorted(u, key=lambda x:x[1])
        u = u[:int(certainty*len(u))] # select the first X % eids with the lowest uncertainty (highest certainty)
        return set([eid for eid,_ in u])



    def kappa_interval_consistency(self):
        ic_agreements, p_agreements = 0, 0
        a1_labs = {}
        a2_labs = {}
        for eid in self.shared_ids:
            (xsl,xsu,xel,xeu) = self.A1s[eid].s_lower.in_minutes_from_Christ(), self.A1s[eid].s_upper.in_minutes_from_Christ(), self.A1s[eid].e_lower.in_minutes_from_Christ(), self.A1s[eid].e_upper.in_minutes_from_Christ()
            (ysl,ysu,yel,yeu) = self.A2s[eid].s_lower.in_minutes_from_Christ(), self.A2s[eid].s_upper.in_minutes_from_Christ(), self.A2s[eid].e_lower.in_minutes_from_Christ(), self.A2s[eid].e_upper.in_minutes_from_Christ()
            lab1 = (xsl,xsu,xel,xeu)
            lab2 = (ysl,ysu,yel,yeu)
            if not lab1 in a1_labs:
                a1_labs[lab1] = 0
            if not lab2 in a2_labs:
                a2_labs[lab2] = 0
            a1_labs[lab1] += 1
            a2_labs[lab2] += 1
            s_agreement = (ysl <= xsl and xsl <= ysu) or (
                        xsl <= ysl and ysl <= xsu)  # (ysl <= xsl <= ysu) or (xsl <= ysl <= xsu)
            e_agreement = (yel <= xel and xel <= yeu) or (
                        xel <= yel and yel <= xeu)  # (yel <= xel <= yeu) or (xel <= yel <= xeu)
            ic_agreement = s_agreement and e_agreement
            p_agreement = s_agreement or e_agreement
            p_agreements += p_agreement
            ic_agreements += ic_agreement

        Po_ic = ic_agreements / len(self.shared_ids)
        Po_p = p_agreements / len(self.shared_ids)

        # iterate through non-0 start end probs and check overlap agreement
        Pe_ic, Pe_p = 0, 0
        for l1 in a1_labs:
            (xsl, xsu, xel, xeu) = l1
            for l2 in a2_labs:
                (ysl,ysu,yel,yeu) = l2
                pl1 = a1_labs[l1] / len(self.shared_ids)
                pl2 = a2_labs[l2] / len(self.shared_ids)


                s_agreement = (ysl <= xsl and xsl <= ysu) or (
                            xsl <= ysl and ysl <= xsu)  # (ysl <= xsl <= ysu) or (xsl <= ysl <= xsu)
                e_agreement = (yel <= xel and xel <= yeu) or (
                            xel <= yel and yel <= xeu)  # (yel <= xel <= yeu) or (xel <= yel <= xeu)
                ic_agreement = s_agreement and e_agreement
                p_agreement = s_agreement or e_agreement

                Pe_ic += (pl1 * pl2 * ic_agreement)
                Pe_p += (pl1 * pl2 * p_agreement)
        kappa_ic = (Po_ic - Pe_ic) / (1 - Pe_ic)
        kappa_p = (Po_p - Pe_p) / (1 - Pe_p)

        print('k_ic',round(kappa_ic,4), 'Pe',round(Pe_ic,4), 'Po', round(Po_ic,4))
        print('k_p',round(kappa_p,4), 'Pe',round(Pe_p,4), 'Po', round(Po_p,4))

        return kappa_ic, kappa_p





    def Pe_interval_overlap(self):
        # get probs for each start and end / annotator

        a1_labs = {}
        a2_labs = {}
        for eid in self.shared_ids:
            xs, xe = self.A1s[eid].s.in_minutes_from_Christ(), self.A1s[eid].e.in_minutes_from_Christ()
            ys, ye = self.A2s[eid].s.in_minutes_from_Christ(), self.A2s[eid].e.in_minutes_from_Christ()
            lab1 = (xs,xe)
            lab2 = (ys, ye)
            if not lab1 in a1_labs:
                a1_labs[lab1] = 0
            if not lab2 in a2_labs:
                a2_labs[lab2] = 0
            a1_labs[lab1] += 1
            a2_labs[lab2] += 1


        # iterate through non-0 start end probs and check overlap agreement
        Pe = 0
        for l1 in a1_labs:
            for l2 in a2_labs:

                (xs, xe), (ys,ye) = l1, l2
                pl1 = a1_labs[l1] / len(self.shared_ids)
                pl2 = a2_labs[l2] / len(self.shared_ids)
                agreement = self.get_interval_overlap_proportion(xs, xe, ys, ye, convert=False)
                Pe += (pl1 * pl2 * agreement)

        return Pe


    def Po_interval_overlap(self):
        overlaps = []
        for eid in self.shared_ids:
            s1, e1 = self.A1s[eid].s, self.A1s[eid].e
            s2, e2 = self.A2s[eid].s, self.A2s[eid].e
            overlap = self.get_interval_overlap_proportion(s1, e1, s2, e2)
            overlaps.append(overlap)
        return np.mean(overlaps)

    def get_interval_overlap_proportion(self, xs, xe, ys, ye, convert=True):
        # Jaccard index of two overlapping intervals
        if convert:
            xs, xe, ys, ye = xs.in_minutes_from_Christ(), xe.in_minutes_from_Christ(), ys.in_minutes_from_Christ(), ye.in_minutes_from_Christ()

        mmax = max([xe,ye])
        mmin = min([xs,ys])
        xd = xe-xs
        yd = ye-ys
        intersection = 0
        union = mmax - mmin
        if mmax - mmin < xd + yd: # OVERLAP
            intersection = (xd + yd) - union


        return intersection / union



    def get_order_class_dicts(self, component_function, A_dict):
        pair_to_class = {}
        for e1 in self.shared_ids:
            e1_point = component_function(A_dict[e1])
            for e2 in self.shared_ids:
                e2_point = component_function(A_dict[e2])
                pair = (e1,e2)
                if e1_point < e2_point:
                    cl = "<"
                elif e1_point == e2_point:
                    cl = "="
                elif e1_point > e2_point:
                    cl = ">"
                else:
                    cl = "?"
                pair_to_class[pair]=cl
        return pair_to_class


    def get_duration_order_of_magnitude_confusion_matrix(self):
        labels = ["Y","M","W","D","H","m","0"]
        confusion_matrix_A1_to_A2 = {l1:{l2:0 for l2 in labels} for l1 in labels}
        A1_class_dict = {eid:self.A1s[eid].d.order_of_magnitude() for eid in self.shared_ids}
        A2_class_dict = {eid:self.A2s[eid].d.order_of_magnitude() for eid in self.shared_ids}
        for eid in self.shared_ids:
            a1_cl = A1_class_dict[eid]
            a2_cl = A2_class_dict[eid]
            confusion_matrix_A1_to_A2[a1_cl][a2_cl]+=1
        return confusion_matrix_A1_to_A2

    def get_order_confusion_matrix(self, component_function):
        labels = [">","=","<"]
        events_involved_in_agreement, events_involved_in_disagreement = [], []

        confusion_matrix_A1_to_A2 = {l1:{l2:0 for l2 in labels} for l1 in labels}
        A1_class_dict = self.get_order_class_dicts(component_function, self.A1s)
        A2_class_dict = self.get_order_class_dicts(component_function, self.A2s)
        for pair in A1_class_dict:
            a1_cl = A1_class_dict[pair]
            a2_cl = A2_class_dict[pair]
            e1,e2 = pair
            if a1_cl == a2_cl:
                events_involved_in_agreement += [e1, e2]
            else:
                events_involved_in_disagreement += [e1, e2]
            confusion_matrix_A1_to_A2[a1_cl][a2_cl]+=1


        return confusion_matrix_A1_to_A2, events_involved_in_agreement, events_involved_in_disagreement


    def cohens_kappa_from_confusion_matrix_dict(self, matrix, disagreement_func=lambda x,y: x!=y):
        labels = list(matrix.keys())
        total_sum = 0
        agreement_count = 0
        A1_counts_per_label, A2_counts_per_label = {l:0 for l in labels},{l:0 for l in labels}
        for l1 in labels:
            for l2 in labels:
                v = matrix[l1][l2]
                total_sum += v
                A1_counts_per_label[l1] += v
                A2_counts_per_label[l2] += v
                if not disagreement_func(l1,l2):
                    agreement_count += v
        Po = agreement_count / total_sum
        A1_prob_per_label, A2_prob_per_label = {l:v/total_sum for l,v in A1_counts_per_label.items()}, {l:v/total_sum for l,v in A2_counts_per_label.items()}

        #print('Po',round(Po,3))
        Pe = 0
        for l1 in labels:
            for l2 in labels:
                pe_l1_l2 = A1_prob_per_label[l1]*A2_prob_per_label[l2]
                if not disagreement_func(l1,l2):
                    Pe += pe_l1_l2
        #print('Pe',round(Pe,3))
        kappa = (Po - Pe) / (1 - Pe)
        return kappa, Po, Pe


    def partial_agreement_krippendorf_alpha(self, component_function):
        a1_label_counts, a1_labels = {}, set()
        a2_label_counts, a2_labels = {}, set()

        # TODO: change this function to use a (lower,upper) pair as ONE label
        for eid in self.shared_ids:
            a1_lower, a1_upper = component_function(self.A1s[eid])
            a2_lower, a2_upper = component_function(self.A2s[eid])
            l1,l1_str = (a1_lower,a1_upper), (str(a1_lower),str(a1_upper))
            l2,l2_str = (a2_lower,a2_upper), (str(a2_lower),str(a2_upper))
            if not l1_str in a1_label_counts:
                a1_label_counts[l1_str] = 0
                a1_labels.add(l1)
            if not l2_str in a2_label_counts:
                a2_label_counts[l2_str] = 0
                a2_labels.add(l2)

            a1_label_counts[l1_str]+=1
            a2_label_counts[l2_str]+=1

        De = 0 # Calculating Expected Disagreement
        for la1 in a1_labels:
            la1_str = (str(la1[0]), str(la1[1]))
            a1_lower, a1_upper = la1
            pla1 = a1_label_counts[la1_str] / len(self.shared_ids)
            for la2 in a2_labels:
                a2_lower, a2_upper = la2
                la2_str = (str(la2[0]), str(la2[1]))
                pla2 = a2_label_counts[la2_str] / len(self.shared_ids)
                p = pla1 * pla2
                disagreement = a1_upper < a2_lower or a2_upper < a1_lower

                De += (disagreement * p)

        #print('De',De)
        # Calculating Observed Disagreement
        num_observed_disagreements = 0
        for eid in self.shared_ids:
            a1_lower, a1_upper = component_function(self.A1s[eid])
            a2_lower, a2_upper = component_function(self.A2s[eid])
            disagreement = a1_upper < a2_lower or a2_upper < a1_lower
            #if disagreement:
            #    print(a1_lower, a1_upper, a2_lower, a2_upper, eid)
            num_observed_disagreements += disagreement

        print(num_observed_disagreements, len(self.shared_ids))
        Do = num_observed_disagreements / len(self.shared_ids)
        #print('De',De,'Do',Do)
        alpha = 1 - (Do / De)
        print('>>>>>',alpha, Do, De)
        Po = 1-Do
        Pe = 1-De

        return alpha, Po, Pe







