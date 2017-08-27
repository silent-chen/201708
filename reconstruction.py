# coding=utf-8
import tensorflow as tf
from PIL import Image,ImageFilter
import numpy as np
def imageprepare(argv):
    im = Image.open(argv)
    imout = im.convert('L')
    xsize, ysize = im.size
    if xsize != 28 or ysize != 28:
        imout = imout.resize((28, 28), Image.ANTIALIAS)
        imout.save("return.png", "png")
    arr = []
    for i in range(28):
        for j in range(28):
            pixel = float(1.0 - float(imout.getpixel((j, i))) / 255.0)
            arr.append(pixel)
    return arr
sess = tf.Session()
saver = tf.train.import_meta_graph("./model.ckpt.meta")
saver.restore(sess,'./model.ckpt')
keep_prob=tf.get_default_graph().get_tensor_by_name('dropout/Placeholder:0')
x=tf.get_default_graph().get_tensor_by_name('Placeholder:0')
y_conv=tf.get_default_graph().get_tensor_by_name('fc2/add:0')
prediction = tf.argmax(y_conv, 1)
array = np.array(imageprepare('./1.png'))
y_pre = y_conv.eval(feed_dict={x: [array], keep_prob: 1.0}, session=sess)
prediction1=np.argmax(y_pre,1)
print('The digits in this image is:%d' % prediction1)