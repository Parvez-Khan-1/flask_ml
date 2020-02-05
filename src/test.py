from nltk.tokenize.punkt import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()
tokenizer._params.abbrev_types.add('dr')
text = "Cytology is report is reviewed by Dr. Modi. The patient test <em>small</em><em>cell</em><em>carcinoma</em> is positive"
sentences = tokenizer.tokenize(text)

print(sentences)


