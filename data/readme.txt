The tsv data contains three sets: 
- the training+dev set (132 documents, each annotated by one annotator).
- the test set (37 unique documents, where each document is annotated by at least 2, sometimes 3 annotators, resulting in 80 files).
- the ground truth test set (the 37 test set documents, with merged annotations).

Each file contains one annotated event per row, and separates columns by tabs. The columns have the following meaning:
1. The i2b2 event id (from the i2b2 temporal challenge from 2012).
2. start time: mode
3. start time: lower bound
4. start time: upper bound
5. duration: mode
6. duration: lower bound
7. duration: upper bound
8. end time: mode
9. end time: lower bound
10. end time: upper bound

The event-annotated texts (not provided here) corresponding to these annotations are from the i2b2 temporal challege 2012.
They can be obtained under a data agreement here: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
