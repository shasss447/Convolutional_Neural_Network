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
        if self.filters is None:
            self.filters=np.random.randn(self.num_filters,3,3,d)/9

        output=np.zeros((h-2,w-2,self.num_filters))

        for im_rg,i,j in self.itr_region(input):
            output[i,j]=np.sum(im_rg*self.filters,axis=(1,2,3))
        
        return output
    
        