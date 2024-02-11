import pickle


def get_occur_dec ():
    with open("beging_func.pkl", "rb") as file:
        benign = pickle.load(file)
    with open("mal_func.pkl", "rb") as file:
        mal = pickle.load(file)
    return benign, mal
n_mx_words=40
def gen_bening_dic():
    benign, mal= get_occur_dec()
    only_func=[]
    for i in benign:
        if not (i in mal) and (benign[i] > n_mx_words):
            only_func.append(i)
    with open("only_benign.pkl", "wb") as file:
        pickle.dump(only_func, file)
    return
def gen_mal_dic():
    benign, mal= get_occur_dec()
    only_func=[]
    for i in mal:
        if mal[i]<0.5*n_mx_words:
            continue
        if not (i in benign) or ((i in benign ) and (benign[i] < mal[i])):
            only_func.append(i)
    with open("only_mal20.pkl", "wb") as file:
        pickle.dump(only_func, file)
    return
gen_bening_dic()
gen_mal_dic()
print ("Thanks")
