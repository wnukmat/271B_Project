import tensorflow as tf
from load_data import *


n_classes = 2

x = tf.placeholder('float', [None, 100, 100, 3])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)



def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
	#filter = tf.get_variable('weights', [5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
			'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			'W_fc':tf.Variable(tf.random_normal([25*25*64,1024])),
			'W_fc2':tf.Variable(tf.random_normal([1024,n_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			'b_conv2':tf.Variable(tf.random_normal([64])),
			'b_fc':tf.Variable(tf.random_normal([1024])),
			'b_fc2':tf.Variable(tf.random_normal([n_classes]))}

	conv1 = tf.nn.relu(conv2d( x, weights['W_conv1']) + biases['b_conv1'])
	print 'after conv1: ', conv1.shape
	conv1 = maxpool2d(conv1)
	print 'after maxpool1: ', conv1.shape	    
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	print 'after conv2: ', conv2.shape
	conv2 = maxpool2d(conv2)
	print 'after maxpool2: ', conv2.shape

	fc = tf.reshape(conv2,[-1, 25*25*64])
	print 'after fc1: ', fc.shape
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['W_fc2'])+biases['b_fc2']
	print 'after fc2: ', output.shape
	return output
	#return output

def train_neural_network(x):
	prediction = convolutional_neural_network(x)
	print 'prediction: ',prediction 

	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
    
	hm_epochs = 20
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(training_set.num_examples/training_set.batch_size) + 1):
				epoch_x, epoch_y = training_set.next_batch()
				#print 'sample batch: ', epoch_x.shape
				#print 'label batch: ', epoch_y.shape
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Accuracy:',accuracy.eval({x:training_set.test_samples, y:training_set.test_labels}))

training_set = training_set()
training_set.load_set()
train_neural_network(x)

        

