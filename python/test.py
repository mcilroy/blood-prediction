import numpy as np
import tensorflow as tf
a = tf.Variable(np.array([[0.45, 0.55], [0.1, 0.9]]), dtype=np.float32, name="ha")
b = tf.matmul(a, a)

q = tf.FIFOQueue(2, dtypes=[tf.float32], shapes=[])
enqueue = q.enqueue_many([a[:, 1]])
dequeue = q.dequeue_many(2)
predmatch = tf.less(dequeue, [.6])
selected_items = tf.reshape(tf.where(predmatch), [-1])
found = tf.gather(dequeue, selected_items)

secondqueue = tf.FIFOQueue(6, dtypes=[tf.float32], shapes=[])
enqueue2 = secondqueue.enqueue_many([found])
dequeue2 = secondqueue.dequeue_many(1) # XXX, hardcoded

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(enqueue)  # Fill the first queue
    sess.run(enqueue2) # Filter, push into queue 2
    #print sess.run(dequeue2) # Pop items off of queue2
    blah = sess.graph.get_operation_by_name("ha")
    #print(blah)
    print(sess.run(blah))
    print(sess.run(a))


    #print("w:", w.name)
    #print(sess.run(w))
