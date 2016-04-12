__author__ = 'david_torrejon'





def fredo_dataset(file_imgid_token='./data_6stdpt/DevData/scaleconcept16.teaser_dev_data_textual.scofeat.v20160212',
    file_imgid_doc_id = './data_6stdpt/DevData/scaleconcept16.teaser_dev_id.v20160212/scaleconcept16.teaser_dev.ImgToTextID.txt'):
    """
    ImgToTextID image doc
    """
    file_img_token_correct = open('scaleconcept16.teaser_dev_data_textual.scofeat.v20160212.david', 'w')
    #file_img_token = open()

    list_tuples_ids = {}
    with open(file_imgid_doc_id) as f:
         for line in f:
             k_v = line.rstrip().split(" ")
             list_tuples_ids[k_v[1]] = k_v[0]

    with open(file_imgid_token) as f:
         for line in f:
             to_change = line.split(" ")
             to_change[0] = list_tuples_ids[to_change[0]]
             print to_change
             to_write = " ".join(to_change)
             file_img_token_correct.write(to_write)
