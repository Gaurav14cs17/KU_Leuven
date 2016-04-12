__author__ = 'david_torrejon'

from file_preparation import open_train_datatxt
from prediction import TbirPredict
from text_preparation import TbirText
from redo_dataset import fredo_dataset


#fredo_dataset()

model = TbirText()
docs_tokens, list_files = model.train_model_lda()
#docs_tokens, list_files = model.train_model_tfidf()
#print docs_tokens
is_dev = True


dict_text = open_train_datatxt(is_dev=is_dev)
pred = TbirPredict(dict_text, is_dev=is_dev)
print len(docs_tokens)
for tokens, filename in zip(docs_tokens, list_files):
    print 'predict', filename
    pred.predict_img(tokens, filename_output=filename, nb_imgs=1)

print("\n")
