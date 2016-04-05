__author__ = 'david_torrejon'

from file_preparation import open_train_datatxt
from prediction import TbirPredict

dict_text = open_train_datatxt()
pred = TbirPredict(dict_text)
pred.predict_img("argentina", nb_imgs=5)
