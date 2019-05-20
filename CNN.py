from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import cv2
import numpy
from scipy import ndimage
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle

#Рабочая директория
directory = 'C:\\Users\\Sigl\\Documents\\Python Scripts\\handwritten_text_recognition\\'

#Количество эпох
epochs_no = 10
#Размер партии данных
batch_size = 50
#Количество меток
classes_num = 61
#Названия меток
classTag = ["А","Б","Ц","Ч","Д","Е","Э","Ф","Г","Х","И","Й","К","Л","М","Н",
"О","П","Р","С","Ш","Щ","Т","У","В","Я","Ю","З","Ж","а","б","ц","ч","д","е","э",
"ф","г","х","и","й","к","л","м","ь","н","о","п","р","с","ш","щ","т","ъ","у","в",
"я","ы","ю","з","ж"]

#Извлечь данные и их метки
def extract_data_with_labels():
    smallDir = directory + 'database\\small'
    bigDir = directory + 'database\\big'
    sm = os.listdir(smallDir)
    b = os.listdir(bigDir)
    img = []
    labels = []
    count = 0
    for folder in b:
        images = glob.glob(bigDir+"\\"+folder+"\\*.png")
        for image in images:
            img.append(cv2.imread(image,0))
            labels.append(count)
        count += 1
    for folder in sm:
        images = glob.glob(smallDir+"\\"+folder+"\\*.png")
        for image in images:
            img.append(cv2.imread(image,0))
            labels.append(count)
        count += 1
    return img, labels

#Аугментация данных при помощи вращения и сдвига
def expand_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    j = 0
    for x, y in zip(images, labels):
        j = j + 1
        if j % 100 == 0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))
        expanded_images.append(x)
        expanded_labels.append(y)
        bg_value = numpy.median(x)  
        image = numpy.reshape(x, (-1, 64))

        for i in range(4):
            angle = numpy.random.randint(-15,15,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)
            expanded_images.append(new_img_)
            expanded_labels.append(y)   
    return expanded_images, expanded_labels

#Загрузка данных для распознавания
def load_images_for_prediction():
    f = os.listdir(directory + 'for_prediction')
    for folder in f:
        f2 = os.listdir(directory + 'for_prediction\\'+folder)
        for folder2 in f2:
            images = glob.glob(directory + 'for_prediction\\'+folder+'\\'+folder2+'\\'+'*.png')    
            #images = glob.glob(directory + 'for_prediction\\*.png')
            predic_img = []
            for image in images:
                predic_img.append(cv2.imread(image,0))     
            predic_img = numpy.array(predic_img)
            predic_img = predic_img.astype('float32') 
            predic_img /= 255
            #Надо бы вынести открытие сессии за циклы
            with tf.Session() as sess:
            #Загрузка весов и смещений
                with open('weight_and_bias.pkl', 'rb') as f:
                    w1, b1, w2, b2, w3, b3, w4, b4 = pickle.load(f)               
                sess.run(tf.global_variables_initializer())
                #Присваивание весов и смещений
                sess.run(W_conv1.assign(w1))
                sess.run(b_conv1.assign(b1))
                sess.run(W_conv2.assign(w2))
                sess.run(b_conv2.assign(b2))
                sess.run(W_fc1.assign(w3))
                sess.run(b_fc1.assign(b3))
                sess.run(W_fc2.assign(w4))
                sess.run(b_fc2.assign(b4))
                #print("Model restored.")
                data = predic_img
                #data = load_images_for_prediction()
                X_test = []
                for i in range(len(data)):
                    X_test.append(data[i].ravel())
                X_test = numpy.asarray(X_test)
                output = tf.argmax(y_conv, 1).eval(feed_dict={x: X_test, keep_prob: 1.0})
                for i in range(len(output)):
                    print(classTag[output[i]],end="")
            print(' ',end="")
        print(' ')
    #return predic_img

#Создание переменных для весов
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='w')

#Создание переменных для смещения
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='b')

#Сверточный слой
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Слой подвыборки
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Модель сверточной нейронной сети
###############################################################################    
x = tf.placeholder(tf.float32, shape=[None, 64*64])
y_ = tf.placeholder(tf.float32, shape=[None, classes_num])   

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 64, 64, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, classes_num])
b_fc2 = bias_variable([classes_num])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Потеря
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#Оптимизация
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#Точность
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
###############################################################################

#Загрузить сохраненные веса и смещения или же обучить нейронную сеть заново
restore = True
if not restore:
    img, label = extract_data_with_labels()
    exp_data, exp_labels = expand_training_data(img, label)
    exp_data = numpy.array(exp_data)
    exp_labels = numpy.array(exp_labels)
    with tf.Session() as temp_sess:
        exp_labels = temp_sess.run(tf.one_hot(indices=tf.cast(exp_labels, tf.int32), depth=classes_num))
    data = []
    for i in range(len(exp_data)):
        data.append(exp_data[i].ravel())
    data = numpy.asarray(data)
    #Валидационные данные я так и не использовались
    X_train, X_valid, Y_train, Y_valid = train_test_split(data, exp_labels, test_size = 0.1)
    #Нормализация данных
    X_train = X_train.astype('float32') 
    X_valid = X_valid.astype('float32')
    X_train /= 255
    X_valid /= 255

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #Итерации по эпохам
        for i in range(epochs_no):
            train_accuracy = 0
            train_loss = 0
            batch1 = 0
            end = False
            count = 0
            #Итерации по партиям данных
            while(True):
                count = count + 1
                batch2 = batch1 + batch_size
                if batch2 >= len(X_train):
                    batch2 = len(X_train)
                    end = True
                train_accuracy = train_accuracy + accuracy.eval(feed_dict={x: X_train[batch1:batch2], y_: Y_train[batch1:batch2], keep_prob: 1.0})
                train_loss = train_loss + cross_entropy.eval(feed_dict={x: X_train[batch1:batch2], y_: Y_train[batch1:batch2], keep_prob: 1.0})
                train_step.run(feed_dict={x: X_train[batch1:batch2], y_: Y_train[batch1:batch2], keep_prob: 0.5})
                batch1 = batch2
                if end:
                    break
            print('Epoch', i + 1, 'completed out of', epochs_no)
            print('Training loss: ',  train_loss / count)
            print('Training accuracy: ',  train_accuracy / count)
            #Сохранение весов и смещений
            with open('weight_and_bias.pkl', 'wb') as f:
                pickle.dump([W_conv1.eval(), b_conv1.eval(), W_conv2.eval(), b_conv2.eval(), W_fc1.eval(), b_fc1.eval(), W_fc2.eval(), b_fc2.eval()], f)
        
else:
    load_images_for_prediction()
# =============================================================================
#     with tf.Session() as sess:
#         #Загрузка весов и смещений
#         with open('weight_and_bias.pkl', 'rb') as f:
#             w1, b1, w2, b2, w3, b3, w4, b4 = pickle.load(f)               
#         sess.run(tf.global_variables_initializer())
#         #Присваивание весов и смещений
#         sess.run(W_conv1.assign(w1))
#         sess.run(b_conv1.assign(b1))
#         sess.run(W_conv2.assign(w2))
#         sess.run(b_conv2.assign(b2))
#         sess.run(W_fc1.assign(w3))
#         sess.run(b_fc1.assign(b3))
#         sess.run(W_fc2.assign(w4))
#         sess.run(b_fc2.assign(b4))
#         print("Model restored.")
#         data = load_images_for_prediction()
#         X_test = []
#         for i in range(len(data)):
#             X_test.append(data[i].ravel())
#         X_test = numpy.asarray(X_test)
#         output = tf.argmax(y_conv, 1).eval(feed_dict={x: X_test, keep_prob: 1.0})
#         print(output)
# =============================================================================
