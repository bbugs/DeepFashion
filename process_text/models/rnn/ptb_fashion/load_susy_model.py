import tensorflow as tf

# Create some variables.
v1 = tf.Variable(tf.random_normal([3,4]), name="v1")
v2 = tf.Variable(tf.random_normal([3,5]), name="v2")
# ...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print "Model restored."
  # Do some work with the model
  # ...
  tvs = tf.trainable_variables()

  print " "

# write a text generation. probably with a forward pass?


