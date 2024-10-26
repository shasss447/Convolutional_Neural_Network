import numpy as np

class Softmax:
    def __init__(self,in_len,n):
        self.wt=np.random.randn(in_len,n)/in_len
        self.b=np.zeros(n)

    def forward(self,input):
        self.lis=input.shape

        input=input.flatten()
        self.li=input

        ts=np.dot(input,self.wt)+self.b
        self.lt=ts

        exp=np.exp(ts)
        return exp/np.sum(exp,axis=0)
    
    def backprop(self,dl_dout,lr):
        for i,g in enumerate(dl_dout):
            if g==0:
                continue

            t_ex=np.exp(self.lt)
            s=np.sum(t_ex)

            dout_dt=-t_ex[i]*t_ex/(s**2)
            dout_dt[i]=t_ex[i]*(s-t_ex[i])/(s**2)
            dt_dw=self.li
            dt_di=self.wt
            dt_db=1
            dl_dt=self.wtdl_dt=g*dout_dt
            dl_dw=dt_dw[np.newaxis].T @ dl_dt[np.newaxis]
            dl_db=dl_dt*dt_db
            dl_di=dt_di@dl_dt

            self.wt-=lr*dl_dw
            self.b-=lr*dl_db

            return dl_di.reshape(self.lis)
        