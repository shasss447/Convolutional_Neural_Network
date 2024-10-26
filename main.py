import tensorflow as tf
import numpy as np
from convolution import Conv3x3
from max_pool import MaxPool2
from softmax import Softmax

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()[:1000]
test_images = np.expand_dims(test_images, axis=-1)

conv=Conv3x3(8)
pool=MaxPool2()
stmx=Softmax(13*13*8,10)

def forward(image,label):
    out=conv.forward((image/255)-0.5)
    out=pool.forward(out)
    out=stmx.forward(out)

    loss=-np.log(out[label])
    acc=1 if np.argmax(out)==label else 0

    return out,loss,acc

def train(label, out, lr=0.005):
    grad=np.zeros(10)
    grad[label]=-1/out[label]
    
    grad=stmx.backprop(grad,lr)

    return loss,acc

loss=0
num_c=0
for i, (im,label) in enumerate (zip(test_images,test_labels)):
    out,l,acc=forward(im,label)
    loss+=l
    num_c+=acc
    
    if i%100==99:
        print(i+1,loss/100,num_c)
        loss=0
        num_c=0

    l,acc=train(label,out)
    l+=1
    num_c+=acc
