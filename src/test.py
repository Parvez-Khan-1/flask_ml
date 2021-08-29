"""
Its a text module
"""
from nltk.tokenize.punkt import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()
tokenizer._params.abbrev_types.add('dr')
TEXT = "Cytology is report is reviewed by Dr. Modi. The patient test " \
       "<em>small</em><em>cell</em><em>carcinoma</em> is positive"
sentences = tokenizer.tokenize(TEXT)
print(sentences)
