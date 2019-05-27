# 杂项


def range_mapping(x, x_min, x_max, y_min, y_max):
    # 值域映射
    y = (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
    return y


def my_fit(sess, x, y, loss_op, optim_op, acc_op, epoch, batch_size, x_train, y_train, x_test, y_test, print_freq=5, eval_train=True):
    import time
    # epoch = 500
    train_x_data = x_train
    train_y_data = y_train
    train_data_len = len(train_x_data)
    test_x_data = x_test
    test_y_data = y_test
    test_data_len = len(test_x_data)
    # batch_size = 30000
    train_batch_count = train_data_len // batch_size + (1 if train_data_len % batch_size != 0 else 0)
    test_batch_count = test_data_len // batch_size + (1 if test_data_len % batch_size != 0 else 0)
    # print_freq = 5
    # eval_train = True
    # need_shuffle = False

    time_train_begin = time.time()
    time_each_epoch_avg = 0
    time_each_epoch_count = 0
    best_acc = 0
    best_acc_epoch = 0
    best_loss = 99999999999
    best_loss_epoch = 0
    for e in range(epoch):
        time_each_epoch_begin = time.time()
        for b in range(train_batch_count):
            batch_start = b * batch_size
            batch_end = batch_start + batch_size
            feed_dict = {x: train_x_data[batch_start:batch_end], y: train_y_data[batch_start:batch_end]}
            los, _ = sess.run([loss_op, optim_op], feed_dict=feed_dict)
        time_each_epoch = time.time() - time_each_epoch_begin
        time_each_epoch_avg += time_each_epoch
        time_each_epoch_count += 1
        if (e + 1) % print_freq == 0 or e + 1 == 1 or e == epoch:
            print('Epoch %d of %d took %fs' % (e + 1, epoch, time_each_epoch_avg / time_each_epoch_count))
            time_each_epoch_avg = 0
            time_each_epoch_count = 0
            if eval_train == True:
                loss_result = 0
                acc_result = 0
                for b in range(train_batch_count):
                    batch_start = b * batch_size
                    batch_end = batch_start + batch_size
                    feed_dict = {x: train_x_data[batch_start:batch_end], y: train_y_data[batch_start:batch_end]}
                    los, ac = sess.run([loss_op, acc_op], feed_dict=feed_dict)
                    loss_result += los
                    acc_result += ac
                loss_result /= train_batch_count
                acc_result /= train_batch_count
                if acc_result > best_acc:
                    best_acc = acc_result
                    best_acc_epoch = e
                if loss_result < best_loss:
                    best_loss = loss_result
                    best_loss_epoch = e
                print("   train loss: %f" % (loss_result))
                print("   train acc: %f" % (acc_result))
            loss_result = 0
            acc_result = 0
            for b in range(test_batch_count):
                batch_start = b * batch_size
                batch_end = batch_start + batch_size
                feed_dict = {x: test_x_data[batch_start:batch_end], y: test_y_data[batch_start:batch_end]}
                los, ac = sess.run([loss_op, acc_op], feed_dict=feed_dict)
                loss_result += los
                acc_result += ac
            loss_result /= test_batch_count
            acc_result /= test_batch_count
            print("   test loss: %f" % (loss_result))
            print("   test acc: %f" % (acc_result))
    print('Total training time: %fs' % (time.time() - time_train_begin))


def get_auto_grow_session():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return tf.Session(config=config)


def enable_auto_grow_eager_execution():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.enable_eager_execution(config)


def tf_force_use_cpu():
    #关闭gpu加速
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
