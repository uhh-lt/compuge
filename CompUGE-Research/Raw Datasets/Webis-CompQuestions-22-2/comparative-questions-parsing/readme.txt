###########

Datasets used for parsing comparative questions by identifying comparison objects, aspects, and predicates.

Datasets:
full.tsv contains all types of comparative questions.
direct.tsv and indirect.tsv contain direct and indirect comparative questions respectively (only object identification).
aspect.tsv contains only comparative questions with aspects (only aspect identification).

########### 

The dataset structure:

sentence_id: id of a respective question.
word:        word tokens.
labels:      token-level labels: OBJ, ASP, PRED for the comparison objects, aspects, and predicates, O none of them.

labels in direct.tsv and indirect.tsv:      token-level labels: OBJ for the comparison objects, O none.
labels aspect.tsv:                          token-level labels: ASP for the comparison aspect, O none.
