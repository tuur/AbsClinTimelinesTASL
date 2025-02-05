# Absolute Event Timeline Annotation Tool
This annotation scheme aims to capture temporal information present in text. It is annotated on top of given event mentions for which the temporal information is to be provided. It answers two questions: (1) what is the most likely time of each event in the text (absolute start and end calender points)? And (2), as very often temporal information in text is ambiguous or underspecified and there is uncertainty for the reader on the exact event time, within what bounds could the event have happened based on the text and the annotators background knowledge?

## What is included in this package?
- A small sample of 1 annotated clinical document in Dutch (and an unannotated English translation). 
- The most recent annotation guidline
- The most recent version of the annotation tool (to visualize and inspect the annotations)

## How to get started with the tool?
The annotation tool is implemented in Python 3. To setup the tool you can run:
```
pip install requirements.txt
```
To run the actual tool:
```
python3 start.py
```
Then the tool should open, and import all .xml files placed in the *data* directory and any subdirectories of that. 
**Important:** Notice that when saving new annotations, these are directly written into the .xml files.

## Starting from raw text?
If you plan to annotate new data starting from raw text, before using this timeline annotation tool, event mentions should be annotated.
As this tool uses the xml format of the i2b2 2012 temporal challenge, the corresponding event annotation tool can be used:
https://github.com/keighrim/mae-annotation (we used v0.9.3).

## Questions?
Feel free to email questions and remarks !

## Reference
When using the tool for research resulting in a publication please reference the following article:
> A. Leeuwenberg & M.-F. Moens (2020), *Towards Extracting Absolute Event Timelines From English Clinical Reports*, IEEE/ACM Transactions on Audio, Speech, and Language Processing.

