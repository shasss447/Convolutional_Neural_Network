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
        h,w,nf=input.shape
        output=np.zeros((h//2,w//2,nf))

        for im_rg,i,j in self.itr_rg(input):
            output[i,j]=np.amax(im_rg,axis=(0,1))

        return output

        