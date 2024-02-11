import pickle
from random import shuffle
from  numpy.random import random as rnd
from create_binar_features import geneate_tensors_for_single_tx
from enum import Enum
class TRANS_MANN(Enum):
    BENUGN = 0
    MALIC = 1
mal_train_por = 0.8
def prepare_malicious():
    list_of_trans = []
    with open("/home/ubuntu/data/roma_mal_trans.pkl", "rb") as file:
        x = pickle.load(file)
        a=1
        for key in x:
            if x[key][2]==False:
                continue
            list_of_trans.append((key,TRANS_MANN.MALIC.value))
        shuffle(list_of_trans)
        with open("list_of_malfiles.pkl", "wb") as file:
              pickle.dump(list_of_trans,file)

        return

def prepare_comon():

    list_of_trans=[]
    with open("list_of_good_files.pkl", "rb") as file:
        x = pickle.load(file)
        a=1
        for key in x:
            if x[key][0]==False:
                continue
            list_of_trans.append((key,TRANS_MANN.BENUGN.value))
        shuffle(list_of_trans)

        lrnds = rnd(len(list_of_trans))
        train_list =[]
        test_list = []
        for j,index in enumerate(list_of_trans):
            if lrnds[j]<mal_train_por:
                train_list.append(index)
            else:
                test_list.append(index)

        with open(ut0.embed_path+"/comm_train.pkl", "wb") as file:
              pickle.dump(train_list,file)
        print (len(train_list))
        with open(ut0.embed_path+"/mal_data.pkl", "rb") as file:
              ml_tx=pickle.load(file)
        tot_data =ml_tx+test_list
        shuffle(tot_data)
        print (len(tot_data))
        with open(ut0.embed_path+"/comm_test.pkl", "wb") as file:
            pickle.dump(tot_data, file)

        return train_list ,test_list

def per_set_of_trans(name_of_file,name_of_trans,outpath,label, ind0,ind1,list_of_topics):
    with open(name_of_file, "rb") as file:
        raw_train = pickle.load(file)

    with open(name_of_trans , "rb") as file:
        x = pickle.load(file)
        for key in raw_train:
            key_val, label0 = key

            if label0 == label:
                geneate_tensors_for_single_tx(x[key_val], ind0,ind1, label, list_of_topics,outpath + key_val+'_'+str(label)+'_.pt')
        return

def   generate_feature():
    with open("uniuqe_files_words.pkl", "rb") as file:
        topics0 = pickle.load(file)
    per_set_of_trans("comm_train.pkl", "filenameb.pkl", "train_folder",TRANS_MANN.BENUGN.value,1,2,topics0)
    per_set_of_trans("comm_test.pkl", "filenameb2.pkl","test_folder",TRANS_MANN.BENUGN.value,1,2,topics0)
    per_set_of_trans("comm_test.pkl", "maltrnas.pkl","test_folder",TRANS_MANN.MALIC.value,1,0,topics0)
    return

if __name__ =='__main__':
    # with open(ut0.embed_path + "/comm_test.pkl", "rb") as file:
    #    x=pickle.load( file)
    a=1
    # prepare_malicious()
    #
    # prepare_comon()
    generate_feature()
    print ("ok")