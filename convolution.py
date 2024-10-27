import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = None
    
    def itr_region(self, image):
        h,w,_=image.shape
        for i in range(h-2):
            for j in range(w-2):
                im_rg=image[i:(i+3),j:(j+3)]
                yield im_rg,i,j
    
    def forward(self, input):
        h,w,d=input.shape
        self.lin=input
        if self.filters is None:
            self.filters=np.random.randn(self.num_filters,3,3,d)/9

        output=np.zeros((h-2,w-2,self.num_filters))

        for im_rg,i,j in self.itr_region(input):
            output[i,j]=np.sum(im_rg*self.filters,axis=(1,2,3))
        
        return output
    
    def backprop(self,dl_dout,lr):
        dl_df=np.zeros(self.filters.shape)
        dl_din=np.zeros(self.lin.shape)

        for im_rg,i,j in self.itr_region(self.lin):
            for f in range(self.num_filters):
                if i < dl_dout.shape[0] and j < dl_dout.shape[1] and f < dl_dout.shape[2]:
                 dl_df[f]+=dl_dout[i,j,f]*im_rg
                 dl_din[i:i+3,j:j+3]+=dl_dout[i,j,f]*self.filters[f]
        
        self.filters-=lr*dl_df

        return dl_din
    
        