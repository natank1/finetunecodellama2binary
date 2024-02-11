import os
import  pickle
import numpy as np

from xgboost import plot_importance,XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from enum import Enum
class TRANS_MANN(Enum):
    BENUGN = 0
    MALIC = 1

def calc_confusion(y_true,y_p):
    # mm = confusion_matrix(y_p, y_true)
    # print(len(y_true), len(y_p))
    # diag = mm[0, 0] + mm[1, 1]
    # diag_s = mm[0, 1] + mm[1, 0]
    # acc = diag / (diag + diag_s)
    # print(acc, diag, diag_s)
    print(confusion_matrix(   y_true,y_p))
    print ("Accuracy=",accuracy_score(y_p, y_true))
    print("Precision mac=", precision_score(y_true,y_p,average='macro'))
    print("Precision mic=", precision_score(y_true, y_p, average='micro'))
    # print ("Geenr F1:",calc_f1(y_true,y_p))
    # print("Geenr Matt:", calc_matt(y_true, y_p))

    return
def construct_new_stack(path0,rndflg=False):
    data_list = os.listdir(path0)
    data_list =[i for i in data_list if i.endswith('.npy')]
    stack_train_all = np.empty(shape=[1,18])
    # stack_y = np.empty(shape=(1,))

    for j,file_name in enumerate(data_list):
        # print (file_name,j)

        np_data  = np.load(path0+file_name, allow_pickle=True)
        a=1
        # if not(np_data[-1]   in [0,1]):
        #     print (file_name, j,np_data)
        # if not( np.argmax(np_data) in [8,9,12]):
        #     print ("yayayay",np.argmax(np_data),np_data)
        # print (max(np_data),np.argmax(np_data))
        # if not (np.argmax(np_data) ==0):
        #     print ('yayay',np_data)
        if rndflg and (int(np_data[-1])==TRANS_MANN.BENUGN.value ) and np.random.rand()>0.13:
            continue
        stack_train_all = np.concatenate((stack_train_all, np.expand_dims(np_data,axis=0)), axis=0)

    # stack_train_all =stack_train_all[1:,:]


    stack_train_all = stack_train_all[1:, :]
    np.random.shuffle(stack_train_all)
    stack_x = stack_train_all[:,:-1]
    # stack_x = stack_train_all[:, :13]

    stack_x= np.concatenate((stack_x, np.expand_dims(np.linalg.norm(stack_x[:,:13], axis=1), axis=1)), axis=1)

    stack_y =np.squeeze(stack_train_all[:,-1])
    stack_x =stack_x[:,[j for j in range(15)]+[17]]
    print (stack_y.shape, stack_x.shape, sum([i for i in  stack_y ]))

    return stack_x,stack_y.astype(int)

if __name__ == '__main__':
    stack_x, stack_y = construct_new_stack("training_features_folder")


    model = XGBClassifier(importance_type='gain', n_estimators=100, learning_rate=0.1)
    model.fit(stack_x, stack_y)
    stack_x =[]
    stack_y =[]
    print ('test')

    for l in range(1):
        test_x, test_y = construct_new_stack("test_features_folder",rndflg=True)
        y_p = model.predict(test_x)
        calc_confusion(   test_y,y_p)
        scores= model.predict_proba(test_x)
        scores =scores[:,1]
        lr_precision, lr_recall, _ = precision_recall_curve(test_y, scores)
        no_skill = len(test_y[test_y == 1]) / len(test_y)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        plt.title("With4Bytes Features")
        # show the plot
        plt.show()
        print ("ok cool")

        a=1
    # zz=plot_importance(model)

    with open('featrue_names.pkl', 'rb') as file:
        names= pickle.load( file)
        ss= names[:15]

        ss.append('norm')
        # ss.extend(names[14:18])
        ss.append('norm')
    print (names)
    mm = {j:i for j,i in zip (ss,model.feature_importances_) }
    mm ={k: v for k, v in sorted(mm.items(), key=lambda item: item[1])}
    plt.barh([i for i in mm], [i for i in mm.values()])
    plt.title("Feature Improtance Including 4bytes")


    plt.show()
