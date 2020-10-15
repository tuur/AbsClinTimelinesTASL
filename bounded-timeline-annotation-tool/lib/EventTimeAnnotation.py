import re, datetime
import numpy as np
from dateutil.relativedelta import *

class CalenderDuration:

    def __init__(self, years=0, months=0, days=0, hours=0, minutes=0):
        self.duration = relativedelta(years=+years, months=+months,days=days, hours=+hours,minutes=+minutes)

    def __eq__(self, other):
        return self.in_minutes() == other.in_minutes()

    def __lt__(self, other):
        return self.in_minutes() < other.in_minutes()

    def __le__(self, other):
        return self < other or self == other

    def __str__(self):
        string = ''
        string+=str(self.duration.years)+'Y'
        string+=str(self.duration.months)+'M'
        string+=str(self.duration.days)+'D'
        string+=str(self.duration.hours)+'H'
        string+=str(self.duration.minutes)+'m'
        return string

    def read_from_string(duration_string):
        regex = r'^(\d+y)(\d+m)(\d+d)(\d+h)(\d+m)$'
        m = re.match(regex, duration_string.lower())
        if m == None:
            print('ERROR: duration string <',duration_string,'> not interpretable.')
        values = [int(v[:-1]) for v in m.groups()]
        return CalenderDuration(years=values[0], months=values[1], days=values[2], hours=values[3], minutes=values[4])

    def as_list_of_strings(self):
        s = [str(self.duration.years), str(self.duration.months), str(self.duration.days), str(self.duration.hours), str(self.duration.minutes)]
        return s

    def as_timedelta(self):
        return self.duration

    def in_minutes(self):
        return abs(self.duration.years*365*24*60) + abs(self.duration.months*31*24*60) + abs(self.duration.days*24*60) + abs(self.duration.hours*60) + abs(self.duration.minutes)

    def order_of_magnitude(self, mins=False):
        if not mins:
            mins = self.in_minutes()

        if mins > 525600:
            return "Y"
        elif mins > 43800:
            return "M"
        elif mins > 10080:
            return 'W'
        elif mins > 1440:
            return "D"
        elif mins > 60:
            return "H"
        elif mins > 0:
            return "m"
        elif mins == 0:
            return "0"


class CalenderPoint:

    def __init__(self, year, month, day, hour, minute):
        self.point = datetime.datetime(year, month, day, hour, minute)

    def __str__(self):
        return self.point.strftime("%Y-%m-%d %H:%M")

    def read_from_string(string):
        point = datetime.datetime.strptime(string, "%Y-%m-%d %H:%M")
        return CalenderPoint(point.year, point.month, point.day, point.hour, point.minute)

    def in_minutes_from_Christ(self):
        datetime_duration_since_christ = self.point - datetime.datetime(1,1,1,1,1)
        dur = CalenderDuration(years=0, months=0, days=datetime_duration_since_christ.days, hours=0, minutes=datetime_duration_since_christ.seconds//60)
        return dur.in_minutes()

    def __eq__(self, other):
        return self.point == other.point

    def __lt__(self, other):
        return self.point < other.point

    def __le__(self, other):
        return self < other or self == other



class EventTimeAnnotation:
    dmin=CalenderDuration(years=0, months=0, days=0, hours=0, minutes=1)

    def __init__(self, event_id, event_string='event', doc_id='doc'):
        self.event_id, self.event_string, self.doc_id = event_id, event_string, doc_id
        self.s, self.s_lower,self.s_upper = None, None, None
        self.e, self.e_lower, self.e_upper = None, None, None
        self.d, self.d_lower, self.d_upper = None, None, None
        self.annotated = []
        self.inferred = []

    def is_complete(self):
        return len(self.annotated) == 2

    def has_the_same_start(self, ann2):
        return str(self.s.point) == str(ann2.s.point) and str(self.s_lower.point)==str(ann2.s_lower.point) and str(self.s_upper.point) == str(ann2.s_upper.point)

    def has_the_same_duration(self, ann2):
        return self.d.in_minutes() == ann2.d.in_minutes() and self.d_lower.in_minutes()==ann2.d_lower.in_minutes() and self.d_upper.in_minutes()== ann2.d_upper.in_minutes()

    def has_the_same_end(self, ann2):
        return str(self.e.point) == str(ann2.e.point) and str(self.e_lower.point)==str(ann2.e_lower.point) and str(self.e_upper.point) == str(ann2.e_upper.point)

    def has_the_same_values(self, ann2):
        same_start = self.has_the_same_start(ann2)
        same_duration = self.has_the_same_duration(ann2)
        same_end = self.has_the_same_end(ann2)
        return same_start and same_duration and same_end

    def is_complete(self):
        for v in [self.s, self.s_lower,self.s_upper,self.e, self.e_lower, self.e_upper,self.d, self.d_lower, self.d_upper]:
            if not v:
                return False
        return True

    def has_close_duration_boundaries(self, f=.2):
        if self.d:
            lower_uncertainty = abs(self.d_lower.in_minutes() - self.d.in_minutes())
            upper_uncertainty = abs(self.d_upper.in_minutes() - self.d.in_minutes())
            if  lower_uncertainty < f * self.d.in_minutes() or upper_uncertainty < f * self.d.in_minutes():
                print(lower_uncertainty, upper_uncertainty, self.d.in_minutes())
                return True
        return False


    def mult_u(self, factor):
        self.d_lower.duration = self.d_lower.duration / factor
        self.d_upper.duration = self.d_upper.duration * factor
        self.s_lower.point = self.s.point - ((self.s.point - self.s_lower.point) * factor)
        self.s_upper.point = self.s.point + ((self.s_upper.point - self.s.point) * factor)
        self.e_lower.point = self.e.point - ((self.e.point - self.e_lower.point) * factor)
        self.e_upper.point = self.e.point + ((self.e_upper.point - self.e.point) * factor)
        return self

    def u_s_lower(self):
        return abs(self.s.in_minutes_from_Christ() - self.s_lower.in_minutes_from_Christ())

    def u_s_upper(self):
        return abs(self.s.in_minutes_from_Christ() - self.s_upper.in_minutes_from_Christ())

    def u_s(self):
        return np.mean([self.u_s_lower(), self.u_s_upper()])

    def u_e_lower(self):
        return abs(self.e.in_minutes_from_Christ() - self.e_lower.in_minutes_from_Christ())

    def u_e_upper(self):
        return abs(self.e.in_minutes_from_Christ() - self.e_upper.in_minutes_from_Christ())

    def u_e(self):
        return np.mean([self.u_e_lower(), self.u_e_upper()])


    def u(self):
        return np.mean([self.u_s(), self.u_e()])

    def has_exact_boundaries(self):
        if self.s and 'start' in self.annotated:
            exact_start = self.s.point == self.s_lower.point or self.s.point == self.s_upper.point
            if exact_start:
                return True
        if self.d and 'duration' in self.annotated:
            exact_duration = self.d.in_minutes() == self.d_lower.in_minutes() or self.d.in_minutes()==self.d_upper.in_minutes()
            if exact_duration:
                return True
        if self.e and 'end' in self.annotated:
            exact_end = self.e.point == self.e_lower.point or self.e.point == self.e_upper.point
            if exact_end:
                return True
        return False

    def is_consistent(self):
        if self.s and 'start' in self.annotated:
            start_consistent = self.s_lower <= self.s and self.s <= self.s_upper
            if not start_consistent:
                return False, 'start'
        if self.d and 'duration' in self.annotated:
            duration_consistent = self.d_lower <= self.d and self.d <= self.d_upper
            if not duration_consistent:
                return False, 'duration'
        if self.e and 'end' in self.annotated:
            end_consistent = self.e_lower <= self.e and self.e <= self.e_upper
            if not end_consistent:
                return False, 'end'
        if self.s and self.e and not self.s <= self.e:
            return False, 'start'
        return True, True

    def add_start(self, most_likely_start, lower_bound, upper_bound):
        self.s = most_likely_start
        self.s_lower = lower_bound
        self.s_upper = upper_bound
        self.annotated.append('start')
        if len(self.annotated) == 2:
            self.infer_complete_annotation()

    def add_duration(self, most_likely_duration, lower_bound, upper_bound, allow_zero_duration=True):
        self.d = most_likely_duration
        self.d_lower = lower_bound
        self.d_upper = upper_bound
        if not allow_zero_duration:
            for d in [self.d, self.d_lower, self.d_upper]:
                if d <= CalenderDuration(0,0,0,0,0):
                    d = self.dmin
        self.annotated.append('duration')
        if len(self.annotated) == 2:
            self.infer_complete_annotation()

    def add_end(self, most_likely_end, lower_bound, upper_bound):
        self.e = most_likely_end
        self.e_lower = lower_bound
        self.e_upper = upper_bound
        self.annotated.append('end')
        if len(self.annotated) == 2:
            self.infer_complete_annotation()

    def get_most_likely_start(self):
        if self.s:
            return self.s.point
        else:
            return self.e.point - self.d.as_timedelta()

    def get_most_likely_end(self):
        if self.e:
            return self.e.point
        else:
            return self.s.point + self.d.as_timedelta()

    def get_most_likely_duration(self):
        if self.d:
            return self.d.as_timedelta()
        else:
            return self.e - self.s


    def resolve_relative_duration(self, anchor):
        dnew = (anchor.point + duration.duration) - anchor.point
        duration.duration = dnew
        return duration

    def resolve_relative_durations(self):
        self.d = self.resolve_relative_duration(self.d, self.s)
        self.d_lower = self.resolve_relative_duration(self.d_lower, self.s)
        self.d_upper = self.resolve_relative_duration(self.d_upper, self.s)


    def infer_complete_annotation(self):

        if len(self.inferred) > 0:
            return self

        if 'start' in self.annotated and 'end' in self.annotated and not 'duration' in self.annotated: # Infer duration from start and end
            time_delta_d, time_delta_d_lower, time_delta_d_upper  = relativedelta(self.e.point, self.s.point), relativedelta(self.e_lower.point, self.s_upper.point), relativedelta(self.e_upper.point, self.s_lower.point)
            self.d = CalenderDuration(time_delta_d.years, time_delta_d.months, time_delta_d.days, time_delta_d.hours, time_delta_d.minutes)
            self.d_lower = CalenderDuration(time_delta_d_lower.years, time_delta_d_lower.months, time_delta_d_lower.days, time_delta_d_lower.hours, time_delta_d_lower.minutes)
            self.d_upper = CalenderDuration(time_delta_d_upper.years, time_delta_d_upper.months, time_delta_d_upper.days, time_delta_d_upper.hours, time_delta_d_upper.minutes)

            if self.d_lower <= CalenderDuration(0,0,0,0,0):
               self.d_lower = self.dmin
            if self.d <= CalenderDuration(0,0,0,0,0):
                self.d = self.dmin

            self.inferred.append('duration')

        elif 'start' in self.annotated and 'duration' in self.annotated and not 'end' in self.annotated: # Infer end from start and duration
            time_e, time_e_lower, time_e_upper = self.s.point + self.d.duration, self.s_lower.point + self.d_lower.duration, self.s_upper.point + self.d_upper.duration
            self.e, self.e_lower, self.e_upper = CalenderPoint(time_e.year, time_e.month, time_e.day, time_e.hour, time_e.minute), CalenderPoint(time_e_lower.year, time_e_lower.month, time_e_lower.day, time_e_lower.hour, time_e_lower.minute), CalenderPoint(time_e_upper.year, time_e_upper.month, time_e_upper.day, time_e_upper.hour, time_e_upper.minute)
            self.inferred.append('duration')

        elif 'duration' in self.annotated and 'end' in self.annotated and not 'start' in self.annotated: # Infer start from duration and end
            time_s, time_s_lower, time_s_upper = self.e.point - self.d.duration, self.e_lower.point - self.d_upper.duration, self.e_upper.point - self.d_lower.duration
            self.s, self.s_lower, self.s_upper = CalenderPoint(time_s.year, time_s.month, time_s.day, time_s.hour, time_s.minute), CalenderPoint(time_s_lower.year, time_s_lower.month, time_s_lower.day, time_s_lower.hour, time_s_lower.minute), CalenderPoint(time_s_upper.year, time_s_upper.month, time_s_upper.day, time_s_upper.hour, time_s_upper.minute)
            self.inferred.append('start')



        return self



if __name__ == "__main__":
    duration = CalenderDuration()
    duration.read_duration_from_string("12Y2M4D0H0m")
    print(duration)

    p = CalenderPoint(2018,10,22,12,0)
    print(p)
    s = str(p)
    np = CalenderPoint.read_from_string(s)
    print(s, np)


def get_annotations_from_event_xml(event_xml, allow_zero_duration=True):
    event_id = event_xml['id']
    if not event_id:
        print("WARNING! No event ID found in:",event_xml)
    ann = EventTimeAnnotation(event_id)

    if event_xml.has_attr('most-likely-duration'):
        d, dmin, dmax = CalenderDuration.read_from_string(
            event_xml['most-likely-duration']), CalenderDuration.read_from_string(
            event_xml['lowerbound-duration']), CalenderDuration.read_from_string(event_xml['upperbound-duration'])
        ann.add_duration(d, dmin, dmax,allow_zero_duration=allow_zero_duration)

    if event_xml.has_attr('most-likely-start'):
        s, smin, smax = CalenderPoint.read_from_string(event_xml['most-likely-start']), CalenderPoint.read_from_string(
            event_xml['lowerbound-start']), CalenderPoint.read_from_string(event_xml['upperbound-start'])
        ann.add_start(s, smin, smax)

    if event_xml.has_attr('most-likely-end'):
        e, emin, emax = CalenderPoint.read_from_string(event_xml['most-likely-end']), CalenderPoint.read_from_string(
            event_xml['lowerbound-end']), CalenderPoint.read_from_string(event_xml['upperbound-end'])
        ann.add_end(e, emin, emax)


    #if not len(ann.annotated) == 2:
    #    print('WARNING:', event_id, '"s annotation is incomplete', event_xml)
    return ann