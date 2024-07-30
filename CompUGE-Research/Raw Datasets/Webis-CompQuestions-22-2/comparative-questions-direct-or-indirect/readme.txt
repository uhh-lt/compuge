###########

Dataset used for classifying comparative questions as direct or indirect.

Datasets:
full.tsv contains an extended dataset with comparative questions labeled as direct or indirect.

########### 

The dataset structure:

id:          id for internal usage in experiments.
direct:      1 if a question is direct, 0 otherwise.
question:    original question.
clean:       cleaned version of a question, punctuation removed.
pos:         question POS tags.
lemma:       question lemmas.
