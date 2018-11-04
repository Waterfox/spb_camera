import tensorflow as tf
import os.path
import scipy.misc
import numpy as np

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1 convolution of vgg layer 7
    layer7a_out = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1,
                                   padding= 'SAME',
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # upsample
    layer4a_in1 = tf.layers.conv2d_transpose(layer7a_out, filters=num_classes, kernel_size=4,
                                             strides= (2, 2),
                                             padding= 'SAME',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1,
                                   padding= 'SAME',
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    ##[8,4,16,22] vs. [8,4,15,21]
    layer4a_out = tf.add(layer4a_in1, layer4a_in2)
    # upsample
    layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, filters=num_classes, kernel_size=4,
                                             strides= (2, 2),
                                             padding= 'SAME',
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # 1x1 convolution of vgg layer 3
    layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1,
                                   padding= 'SAME',
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer3a_out = tf.add(layer3a_in1, layer3a_in2)
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, filters=num_classes, kernel_size=16,
                                               strides= (8, 8),
                                               padding= 'SAME',
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    return nn_last_layer


class beerClassifier(object):
    def __init__(self):
        print('TensorFlow Version: {}'.format(tf.__version__))
        # Check for a GPU
        if not tf.test.gpu_device_name():
            print('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        self.num_classes = 4
        self.image_shape = (160, 320)
        self.vgg_path = "/home/robbie/spb_data/vgg"
        self.data_dir = "/home/robbie/spb_data"
        self.runs_dir = "/home/robbie/spb_data"
        self.training_dir = "/home/robbie/spb_data/data_beer/training"

        with tf.Session() as self.sess:
            #load layers from vgg: input, layer3, layer4, later 7
            self.input_image, self.keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(self.sess, self.vgg_path)
            #create new layers with layers from vgg
            nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, self.num_classes)
            saver = tf.train.Saver()
            self.logits = tf.reshape(nn_last_layer, (-1, self.num_classes))
            ## Restore saved checkpoint ----------------
            saver.restore(self.sess, "/home/robbie/spb_data/models/model.ckpt")
            print("Model restored.")
            self.run()

    def run_classifier(self,image):
        ##check if image has proper shape)
        if image.shape != self.image_shape:
            image = scipy.misc.imresize(image, self.image_shape)

        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)],
            {self.keep_prob: 1.0, self.input_image: [image]})

        #class 1 beer
        im_softmax1 = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
        segmentation1 = (im_softmax1 > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        mask1 = np.dot(segmentation1, np.array([[0, 255, 0, 200]]))
        mask1 = scipy.misc.toimage(mask1, mode="RGBA")

        image_out = scipy.misc.toimage(image)
        image_out.paste(mask1, box=None, mask=mask1)

        # output_dir = os.path.join(self.runs_dir, str(time.time()))
        scipy.misc.imsave(os.path.join(self.data_dir, 'test', '0106_out.png'), image_out)

    def run(self):
        image_file = os.path.join(self.data_dir, 'test', '0106.png')
        image = scipy.misc.imresize(scipy.misc.imread(image_file), (160, 320))
        self.run_classifier(image)

if __name__ == '__main__':
    beerClassifier()
