###########

Dataset used for classifying comparative questions as with an aspect or without an aspect.

Datasets:
full.tsv contains a dataset with comparative questions labeled as with an aspect or without an aspect.

########### 

The dataset structure:

id:          id for internal usage in experiments.
asp:         1 if a question contains an aspect, 0 otherwise.
question:    original question.
clean:       cleaned version of a question, punctuation removed.
pos:         question POS tags.
lemma:       question lemmas.
