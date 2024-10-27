import numpy as np

class MaxPool2:
    def itr_rg(self,image):
        h,w,_=image.shape
        new_h=h//2
        new_w=w//2
        for i in range(new_h):
            for j in range(new_w):
                im_rg=image[i*2:i*2+2,j:j*2+2]
                yield im_rg,i,j
    
    def forward(self,input):
        self.lin=input
        h,w,nf=input.shape
        output=np.zeros((h//2,w//2,nf))

        for im_rg,i,j in self.itr_rg(input):
            output[i,j]=np.amax(im_rg,axis=(0,1))

        return output
    
    def backprop(self,dl_dout):
        dl_din=np.zeros(self.lin.shape)
        for im_rg,i,j in self.itr_rg(self.lin):
            h,w,f=im_rg.shape
            mx=np.amax(im_rg,axis=(0,1))
            for i1 in range(h):
                for j1 in range(w):
                    for k1 in range (f):
                        if im_rg[i1,j1,k1]==mx[k1]:
                            if i*2 + i1 < dl_din.shape[0] and j*2 + j1 < dl_din.shape[1] and k1 < dl_dout.shape[2]:
                             dl_din[i*2+i1,j*2+j1,k1]=dl_dout[i,j,k1]
        
        return dl_din


        