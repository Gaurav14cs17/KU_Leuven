__author__ = 'david_torrejon'

from file_preparation import open_train_datatxt
from prediction import TbirPredict
from text_preparation import TbirText


model = TbirText()
tokens = model.train_model_lda()

is_dev = True

dict_text = open_train_datatxt(is_dev=is_dev)
pred = TbirPredict(dict_text, is_dev=is_dev)
pred.predict_img(tokens, nb_imgs=5)
