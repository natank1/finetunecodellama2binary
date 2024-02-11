import pickle
from enum import Enum
class TRANS_MANN(Enum):
    BENUGN = 0
    MALIC = 1
import numpy as np
from create_binar_features import geneate_feaure_for_single_tx
from web3 import Web3
infurkey="infura"
def per_set_of_trans(w3,name_of_file,name_of_trans,outpath,label, ind0,ind1,list_of_topics,first_time=False):
    with open(name_of_file, "rb") as file:
        raw_train = pickle.load(file)
    flg =False
    if first_time:
        flg=True
    with open(name_of_trans , "rb") as file:
        x = pickle.load(file)
        for key in raw_train:
            if not (key[0] in x):
                continue
            key_val, label0 = key

            if label0 == label:
                if flg : #save names
                    geneate_feaure_for_single_tx(w3,x[key_val], ind0, ind1, label, list_of_topics,outpath + key_val + '_' + str(label) + '_.npy',flg)
                    flg=False
                else:
                    geneate_feaure_for_single_tx(w3,x[key_val], ind0,ind1, label, list_of_topics,outpath + key_val+'_'+str(label)+'_.npy')
        return
def get_w3_object():
    infura_url = 'https://mainnet.infura.io/v3/'+infurkey
    w3 = Web3(Web3.HTTPProvider(infura_url))
    return w3
def   generate_feature():
    with open("uniuqe_words.pkl", "rb") as file:
        topics0 = pickle.load(file)
    w3= get_w3_object()
    per_set_of_trans(w3,"train_data.pkl", "benignfile.pkl", "trainfolder",TRANS_MANN.BENUGN.value,1,2,topics0,first_time=True)
    per_set_of_trans(w3, "train_data.pkl", "all_valid_out.pkl", "trainfolder",TRANS_MANN.BENUGN.value, 1, 0, topics0)

    per_set_of_trans(w3, "train_data.pkl", "all_comb_mal_trans.pkl", "trainfolder",TRANS_MANN.MALIC.value,1,0,topics0)
    per_set_of_trans(w3, "test_data.pkl", "new_good_trans00.pkl", "test_features",TRANS_MANN.BENUGN.value,1,2,topics0)
    per_set_of_trans(w3, "test_data.pkl", "all_valid_out.pkl", "test_features",TRANS_MANN.BENUGN.value, 1, 0, topics0)
    per_set_of_trans(w3, "test_data.pkl", "all_comb_mal_trans.pkl","test_features",TRANS_MANN.MALIC.value,1,0,topics0)
    return




if __name__ =='__main__':
   generate_feature()
   print ("well done")