
import pickle
import textwrap


class octet_process:
    def __init__(self):
        self.path_dic = "folder_path"
        self.mal_dic_f =self.path_dic + "mal_file.pkl"
        self.benign_dic_f= self.path_dic +"benignfilepkl"
        self.mal_dic, self.benign_dic =self.get_cat_dics()
        with open("hexa_file.pkl", "rb") as file:
            self.hex_dic =pickle.load( file)
        return
    def get_cat_dics(self):
        with open(self.mal_dic_f, "rb") as file:
            mal_dic= pickle.load( file)
        with open( self.benign_dic_f, "rb") as file:
            benign_dic =pickle.load( file)
        return mal_dic, benign_dic

    def generate_octet_features(self,tx_input ):
        list_of_ordered_octets = textwrap.fill(tx_input[2:], 8).split()
        n_octets =len(list_of_ordered_octets)
        valid_octet=0
        mal_octet=0
        benign_octet=0
        for octet in list_of_ordered_octets:
            mm = '0x' + octet
            if '0x' + octet in self.hex_dic:
                valid_octet += 1

                if self.hex_dic[mm] in self.mal_dic:
                   mal_octet+=1
                   continue
                if self.hex_dic[mm] in self.benign_dic:
                    benign_octet+=1
                    continue
        return n_octets, valid_octet, benign_octet,mal_octet

    def create_dic(self,tx_input  ):

        ruuning_dic= {}
        list_of_ordered_octets = textwrap.fill(tx_input[2:], 8).split()


        for octet in list_of_ordered_octets:
            mm = '0x' + octet
            if mm in self.hex_dic:
                func0 =self.hex_dic[mm]
                ruuning_dic.setdefault(func0,0)
                ruuning_dic[func0]+=1


        return ruuning_dic

if __name__ =='__main__':
    oct_obj= octet_process()
    a=1

    print(a)