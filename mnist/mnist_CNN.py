# coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import time

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_1_SIZE = 3
CONV1_1_KERNEL_NUM = 16
CONV1_2_SIZE = 3
CONV1_2_KERNEL_NUM = 16
CONV2_1_SIZE = 5
CONV2_1_KERNEL_NUM = 32
CONV2_2_SIZE = 5
CONV2_2_KERNEL_NUM = 32
FC_SIZE = 1000
OUTPUT_NODE = 10

BATCH_SIZE = 25
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.96
epoch = 1
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"


def get_weight(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




def forward(x, train):
    conv1_1_w = get_weight([CONV1_1_SIZE, CONV1_1_SIZE, NUM_CHANNELS, CONV1_1_KERNEL_NUM])
    conv1_1_b = get_bias([CONV1_1_KERNEL_NUM])
    conv1_1 = conv2d(x, conv1_1_w)
    # 采用relu激活函数
    relu1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_b))

    conv1_2_w = get_weight([CONV1_2_SIZE, CONV1_2_SIZE, CONV1_1_KERNEL_NUM, CONV1_2_KERNEL_NUM])
    conv1_2_b = get_bias([CONV1_2_KERNEL_NUM])
    conv1_2 = conv2d(relu1_1, conv1_2_w)
    relu1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_2_b))
    pool1 = max_pool_2x2(relu1_2)

    conv2_1_w = get_weight([CONV2_1_SIZE, CONV2_1_SIZE, CONV1_2_KERNEL_NUM, CONV2_1_KERNEL_NUM])
    conv2_1_b = get_bias([CONV2_1_KERNEL_NUM])
    conv2_1 = conv2d(pool1, conv2_1_w)
    relu2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_b))
    conv2_2_w = get_weight([CONV2_2_SIZE, CONV2_2_SIZE, CONV2_1_KERNEL_NUM, CONV2_2_KERNEL_NUM])
    conv2_2_b = get_bias([CONV2_2_KERNEL_NUM])
    conv2_2 = conv2d(relu2_1, conv2_2_w)
    relu2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_2_b))
    pool2 = max_pool_2x2(relu2_2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE])
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE])
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b

    return y



def from_tensor_slices(x,y,batch):
    r = []
    for i in range(batch,len(y)+1,batch):
        tumple=(x[i-batch:i],y[i-batch:i])
        r.append(tumple)
    return r

def backward():
    x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS])
    x_image= tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32,[None, OUTPUT_NODE])
    y = forward(x_image,True)
    global_step = tf.Variable(0, trainable=False)

    # 函数tf.ragmax(张量名，axis=操作轴)返回特定张量指定维度最大值的索引号，用法：tf.argmax
    # 函数tf.reduce_mean(张量名，axis=操作轴)返回特定张量指定维度的平均值
    # 交叉熵损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(ce)


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/ BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())



    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)


        train_image, train_label = mnist.train.images, mnist.train.labels
        # 实现训练集batch级图片与标签的配对
        # train_db = tf.data.Dataset.from_tensor_slices((train_image, train_label)).batch(BATCH_SIZE)
        train_db = from_tensor_slices(train_image, train_label,BATCH_SIZE)

        # batch = mnist.train.next_batch(BATCH_SIZE)
        for i in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
            for index, (x_train, y_train) in enumerate(train_db):  # batch级别的循环
                _, loss_value, step, acc = sess.run([train_op, loss, global_step,accuracy], feed_dict={x: x_train, y_: y_train})
                if step % 100 == 0:
                    print("After %d training steps, loss on training batch is %g, this batch accuracy is %.4f"%(step, loss_value,acc))
                    saver.save(sess,os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)



def test():

    with tf.Graph().as_default() as g:
        images_test, labels_test = mnist.test.images, mnist.test.labels
        # 随机选取1000张进行准确率测试
        # idx = np.random.choice(mnist.test.num_examples,size=1000,replace=False)
        # xs = images_test[idx, :]
        # ys = labels_test[idx, :]
        xs = images_test
        ys = labels_test

        x = tf.placeholder(tf.float32,[len(images_test),IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS])
        x_image= tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE])
        y = forward(x_image,False)

        saver = tf.train.Saver()

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载训练好的模型参数
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_:ys})
                print("After %d epoch,%s training steps, test accuracy = %g."%(epoch,global_step,accuracy_score))
            else:
                print("No checkpoint file found.")


if __name__ == '__main__':
    # 加载数据集,由于网络原因，多数情况下自动创建数据集会失败，可手动创建MNIST_data文件夹，
    # 并把mnist的四个数据集压缩包下载到此文件夹
    start_time = time.time()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    backward()
    test()
    end_time = time.time()
    time_cost = end_time-start_time
    print("运行时间%d minutes %g seconds"%(int((time_cost-time_cost%60)/60),time_cost%60))
