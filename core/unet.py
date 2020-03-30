from diabetic_package.file_operator.bz_path import get_file_path
import matplotlib.pylab as plt
import cv2
import tensorflow as tf
import numpy as np

train_img_dir = r'D:\Pycharm_Project\mask_rcnn_TF\voc\train\imgs'
train_label_dir = r'D:\Pycharm_Project\mask_rcnn_TF\voc\train\labels'
train_img_list = sorted(get_file_path(train_img_dir, ret_full_path=True))
train_label_list = sorted(get_file_path(train_label_dir, ret_full_path=True))



def generate(train_img_list,train_label_list,batch):
    img_paths = train_img_list[:batch]
    label_path = train_label_list[:batch]
    batch_img=[]
    batch_label=[]
    for img in img_paths:
        img = cv2.imread(img,1)
        img = cv2.resize(img,(512,512))
        img=np.expand_dims(img,axis=0)
        batch_img.append(img)
    batch_img = np.concatenate(batch_img,axis=0)
    for label in label_path:
        label = cv2.imread(label,0)
        label = cv2.resize(label,(512,512))
        label=np.expand_dims(label,axis=[0,-1])
        batch_label.append(label)
    batch_label = np.concatenate(batch_label,axis=0)
    return batch_img,batch_label

class Unet:
    def conv_layer(self, inputs, filters, reuse=tf.get_variable_scope().reuse):
        inputs = tf.layers.conv2d(inputs, filters, 3, padding="same", activation=tf.nn.relu, name='conv1', reuse=reuse)
        inputs = tf.layers.batch_normalization(inputs, name='BN_1')

        inputs = tf.layers.conv2d(inputs, filters, 3, padding="same", activation=tf.nn.relu, name='conv2', reuse=reuse)
        inputs = tf.layers.batch_normalization(inputs, name='BN_2')

        inputs = tf.layers.conv2d(inputs, filters, 3, padding="same", activation=tf.nn.relu, name='conv3', reuse=reuse)
        inputs = tf.layers.batch_normalization(inputs, name='BN_3')

        inputs = tf.layers.max_pooling2d(inputs, 2, 2)
        return inputs

    def deconv_layer(self, inputs, filters, feature, reuse=tf.get_variable_scope().reuse):
        inputs = tf.concat([feature, inputs], axis=-1)
        inputs = tf.layers.conv2d_transpose(inputs, filters, 3, strides=(2, 2), padding='same', activation=tf.nn.relu,
                                            name='deconv1', reuse=reuse)
        inputs = tf.layers.conv2d_transpose(inputs, filters, 3, padding='same', activation=tf.nn.relu,
                                            name='deconv2', reuse=reuse)
        inputs = tf.layers.conv2d_transpose(inputs, filters, 3, padding='same', activation=tf.nn.relu,
                                            name='deconv3', reuse=reuse)
        return inputs

    def build(self, input_image):
        inputs = tf.layers.conv2d(input_image, 64, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 64, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 64, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        C1 = inputs = tf.layers.max_pooling2d(inputs, 2, 2)

        inputs = tf.layers.conv2d(inputs, 128, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 128, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 128, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        C2 = inputs = tf.layers.max_pooling2d(inputs, 2, 2)

        inputs = tf.layers.conv2d(inputs, 256, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 256, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 256, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)
        C3 = inputs = tf.layers.max_pooling2d(inputs, 2, 2)
        inputs = tf.layers.conv2d(inputs, 512, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 512, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)

        inputs = tf.layers.conv2d(inputs, 512, 3, padding="same", activation=tf.nn.relu)
        inputs = tf.layers.batch_normalization(inputs)
        C4 = inputs = tf.layers.max_pooling2d(inputs, 2, 2)

        inputs = tf.layers.conv2d(inputs, 1024, 3, padding='same', activation=tf.nn.relu, name='conv_filter1024')

        inputs = tf.concat([C4, inputs], axis=-1)
        inputs = tf.layers.conv2d_transpose(inputs, 512, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)

        inputs = tf.concat([C3, inputs], axis=-1)
        inputs = tf.layers.conv2d_transpose(inputs, 256, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)

        inputs = tf.concat([C2, inputs], axis=-1)
        inputs = tf.layers.conv2d_transpose(inputs, 128, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)

        inputs = tf.concat([C1, inputs], axis=-1)
        logits = tf.layers.conv2d_transpose(inputs, 64, 3, strides=(2, 2), padding='same', activation=tf.nn.relu)

        return logits


# 定义网络
# def unet(input):
#     input = tf.identity(input, name='input_layer')
#     C1=input = conv_layer(input, 64,name='conv1')
#
#     C2=input = conv_layer(input, 128,name='conv2')
#
#     C3=input = conv_layer(input, 256,name='conv3')
#
#     C4=input = conv_layer(input, 512,name='conv4')
#
#     input = tf.layers.conv2d(input, 1024, 3, padding='same', activation=tf.nn.relu, name='conv_filter1024')
#     # 反卷积
#     with tf.variable_scope('Deconv1'):
#         input = deconv_layer(input, 512,C4,name='deconv1')
#     with tf.variable_scope('Deconv2'):
#         input = deconv_layer(input, 256,C3,name='deconv2')
#     with tf.variable_scope('Deconv3'):
#         input = deconv_layer(input, 128,C2,name='deconv3')
#     with tf.variable_scope('Deconv4'):
#         logits = deconv_layer(input, 21,C1,name='deconv4')
#     return logits


input_tensor = tf.placeholder(tf.float32, [None, 512, 512, 3])

label = tf.placeholder(tf.float32, [None, 512, 512, 1])

logits = Unet().build(input_tensor)
# 构建loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

# 构建评价指标
predictions = tf.nn.softmax(logits)
# metrics = tf.metrics.recall(labels=label,predictions=predictions,name='my_recall')

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            batch_img, batch_label = generate(train_img_list, train_label_list,2)
            sess.run(optimizer, feed_dict={
                input_tensor: batch_img,
                label: batch_label
            })
            loss_ = sess.run([loss], feed_dict={
                input_tensor: batch_img,
                label: batch_label
            })
            print('loss:', loss_)


if __name__ == '__main__':
    import numpy as np

    # input_tensor = tf.placeholder(tf.float32,[None,None,None,3])
    # img = r'D:\Pycharm_Project\mask_rcnn_TF\1.bmp'
    # img = cv2.imread(img, 1)
    # img = cv2.resize(img, (1024, 1024))
    # # img = np.expand_dims(img,0)
    # img = tf.convert_to_tensor(img)
    # img = tf.expand_dims(img, 0)
    # img = tf.cast(img, tf.float32)
    # net = unet(img)
    # net = unet(input_tensor)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     value = sess.run(net,feed_dict={
    #         input_tensor:img
    #     })
    #     for i in range(value.shape[-1]):
    #         value[0,:,:,i]
    train()
