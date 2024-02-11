import torch
import matplotlib.pyplot as plt

class PCA_object:
    def __init__(self, data,ncomp=10 ):
        self.ncomp =ncomp
        self.pca = torch.pca_lowrank(data,self.ncomp )
        return
    def prject_dat(self, curr_data):
       projected_data = torch.matmul(curr_data, self.pca[2])
       return projected_data

    def pl_tensor(self,data0,data1,color0,mrk0,color1,mrk1,title=''  ):
        # plt.plot(data .to('cpu').numpy(),'r')
        fig = plt.figure()
        data0= data0.detach().cpu()
        data1 = data1.detach().cpu()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c=color1, marker=mrk1)

        ax.scatter(data0[:,0],data0[:,1],data0[:,2], c=color0, marker=mrk0)
        plt.legend(["Malicious","Benign"],loc='upper left')
        ax.set_title(title)
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        plt.show()
        return ax# ax.plot_surface(data[:,0],data[:,1],data[:,2], cmap='viridis')


    # def plt_compare(self,data):
    #     ax1=self.pl_tensor( data, 'r', 'o')
    #     ax2 = self.pl_tensor(data+7., 'g', '*')
    #     plt.show()
if __name__ =='__main__':
    a=1
    data = torch.rand(100, 1000)
    pca =PCA_object(data)
    mm =pca.prject_dat(data[:50,:])
    pca.pl_tensor(mm[:,[1,2,3]],'r','+','g','o')
    print (mm.shape)