import tensorflow as tf

filename_queue=tf.train.string_input_producer(['data_for_linear_regression_csv_data_load/test.csv'],shuffle=False,name='filename_queue')
reader=tf.TextLineReader()
key,value=reader.read(filename_queue)

record_defaults=[[0.],[0.],[0.],[0.]]
#t=tf.decode_csv(key,record_defaults=record_defaults)
xy=tf.decode_csv(value,record_defaults=record_defaults)

sess=tf.Session()
for i in range(6):
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    print(sess.run([xy]))
    coord.request_stop()
    coord.join(threads)
