import tensorflow as tf

a = tf.add(3, 5)

sess = tf.Session()
print sess.run(a)

sess.close()

with tf.Session() as sess:
	print sess.run(a)

x = 2
y = 3

op1 = tf.add(x, y)
op2 = tf.add(x, y)

useless = tf.add(x, op1)

op3 = tf.pow(op2, op1)

with tf.Session() as sess:
	print sess.run(op3)

g = tf.Graph()

with g.as_default():
	x = tf.add(3, 5)

sess = tf.Session(graph = g)	# session is run on the graph g

with tf.Session() as sess:
	sess.run(x)

g = tf.get_default_graph()	# handle the default graph

g1 = tf.get_default_graph()
g2 = tf.Graph()	# user created graph

# add ops to the default graph
with g1.as_default():
	a = tf.constant(3)

# add ops to the user created graph
with g2.as_default():
	b = tf.constant(5)