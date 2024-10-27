import tensorflow as tf
import numpy as np
from convolution import Conv3x3
from max_pool import MaxPool2
from softmax import Softmax

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images=train_images[:1000]
train_labels=train_labels[:1000]
test_images=test_images[:1000]
test_labels=test_labels[:1000]

train_images=np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

conv=Conv3x3(16)
pool=MaxPool2()
stmx=Softmax(13*13*16,10)

def forward(image,label):   
    out=conv.forward((image/255)-0.5)
    out=pool.forward(out)
    out=stmx.forward(out)

    loss=-np.log(out[label])
    acc=1 if np.argmax(out)==label else 0

    return out,loss,acc

def train(im,label,lr=0.005):
    out,loss,acc=forward(im,label)
    grad=np.zeros(10)
    grad[label]=-1/out[label]
    
    grad=stmx.backprop(grad,lr)
    grad=pool.backprop(grad)
    grad=conv.backprop(grad,lr)
    return loss,acc

loss=0
num_c=0
for i, (im,label) in enumerate (zip(train_images,train_labels)):
    if i>0 and i%100==99:
        print(i+1,loss/100,num_c)
        loss=0
        num_c=0
    l,acc=train(im,label)
    loss+=l
    num_c+=acc

print('Training done')

loss=0
num_c=0
for im,label in zip(test_images,test_labels):
    _,l,acc=forward(im,label)
    loss+=l
    num_c+=acc
print('Test Loss:',loss/1000)
print('Test Accuracy:',num_c/1000)