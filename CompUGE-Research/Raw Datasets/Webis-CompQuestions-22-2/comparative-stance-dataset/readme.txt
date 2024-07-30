###########

Dataset used for detecting the stance of answers to comparative questions.

The data sources are the Stack Exchange dumps [ https://archive.org/details/stackexchange ] and 
L6 - Yahoo! Answers Comprehensive Questions and Answers version 1.0 (multi part) [ https://webscope.sandbox.yahoo.com/catalog.php?datatype=l ]

Since the Yahoo! Answers data is bound by the non-disclose agreement, we provide the question ids and answer ids for about a half of the dataset entries. We ask you to obtain the dataset from webscope.sandbox.yahoo.com directly and use the notebook process_stance_dataset.ipynb to fetch the questions and answers for the Yahoo part.

########### 

The dataset structure:

ds:                     the data source.
id:                     entry id (not used).
question:               question from Yahoo or Stack Exchange.
answer:                 "best" / "accepted" answer from Yahoo or Stack Exchange.
answer_stance:          0: No stance, 1: Neutral, 2: Pro first object, 3: Pro second object.
answer_stance object:   The object with the pro stance is named.
object_1, object_2:     First / second compared object in the question.
mask_pos_1, mask_pos_2: The list of position of the object_1 / object_2 mentions in the answer (Note: the objects were labeled manually, so that the objects in the answer can be syntactically different from the ones in the question, e.g., synonyms, acronyms, abbreviations, etc.).
