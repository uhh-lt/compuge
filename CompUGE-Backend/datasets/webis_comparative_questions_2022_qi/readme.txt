###########

Datasets used for classifying questions as comparative or not.

The data sources are the questions from MS MARCO, Natural Questions, and Quora Question Pairs.

data_full.tsv contains all questions.
hard_questions.tsv contains the questions that were not classified as comparative by the rules.
very_hard.tsv contains the questions that were not classified as comparative by the logistic regression.

########### 

The dataset structure:

id:       technical id, used in the code for matching examples.
comp:     1 if a question is comparative.
question: original question from the data source.
clean:    lower-cased question, punctuation removed.
pos:      POS tags using Stanza.
lemma:    lemmas using Stanza.
