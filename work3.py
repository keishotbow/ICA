import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np

# mnist数字画像データセット取得
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# パラメータ定義
learning_rate = 0.01  # 学習率
training_epoch = 2  # 学習回数
batch_size = 100  # バッチサイズ
display_step = 1  # ストライド
momentum = 0.9  # モメンタム(慣性項)
probability = 0.5  # ドロップアウトの確率


# モデル生成
def inference(x, keep_prob):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 畳み込み、プール
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)

    # 畳み込み、プール
    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)

    # 全結合
    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7 * 7 * 64, 1024], [1024])

        # ドロップアウト適用
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    # Relu適用
    with tf.variable_scope("output"):
        output = layer(fc_1_drop, [1024, 10], [10])

    return output


# 作成途中
def inference2(x, keep_prob):
    x = tf.reshape(x, shape=[-1, 96, 96, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [3, 3, 32, 32], [32])
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("conv_3"):
        conv_3 = conv2d(pool_2, [3, 3, 32, 64], [64])
        pool_3 = max_pool(conv_3)

    with tf.variable_scope("conv_4"):
        conv_4 = conv2d(pool_3, [5, 5, 64, 64], [64])
        pool_4 = max_pool(conv_4)

    with tf.variable_scope("fc1"):
        pool_4_flat = tf.reshape(pool_4, [-1, ])


# 損失関数(交差エントロピー)の値を算出する
def loss(output, y):
    # elementwise_product = y * tf.log(output)
    # xentropy = -tf.reduce_sum(elementwise_product, reduction_indices=1)

    # 交差エントロピー計算
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=y
    )
    loss = tf.reduce_mean(xentropy)
    return loss


# モデルパラメータの勾配を求め、モデルを更新する
def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    # アダムオプティマイザ適用
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                       use_locking=False, name="Adam")
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op


# モデル性能の評価
def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# 全結合, Relu適用
def layer(input, weight_shape, bias_shape):
    # 正規分布で重み初期化
    weight_init = tf.random_normal_initializer(
        stddev=(2.0 / weight_shape[0]) ** 0.5
    )
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


# 畳み込み層
def conv2d(input, weight_shape, bias_shape):
    tmp_in = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(
        stddev=(2.0 / tmp_in) ** 0.5
    )
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.relu(tf.nn.bias_add(conv_out, b))


#  マックスプーリング
def max_pool(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


# メイン関数
if __name__ == '__main__':

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            # mnistデータ 28*28=784
            x = tf.placeholder('float', [None, 784])
            # 0-9の10個のクラスに分類する
            y = tf.placeholder('float', [None, 10])

            output = inference(x, probability)
            cost = loss(output, y)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = training(cost, global_step)
            eval_op = evaluate(output, y)

            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            if not os.path.exists('logistic_logs'):
                print("学習済みデータを格納しているフォルダを見つけられませんでした")
                exit(1)

            summary_writer = tf.summary.FileWriter(
                "logistic_logs",
                graph_def=sess.graph_def
            )

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # 学習済みモデルが無ければ学習、あれば学習済みデータを使う
            ckpt = tf.train.get_checkpoint_state('logistic_logs')
            if ckpt:
                last_model = ckpt.model_checkpoint_path
                print(last_model)
                # 学習済みデータ読み込み
                # saver.restore(sess, last_model)
                saver.restore(sess, "logistic_logs\model-checkpoint-55000")
            else:
                # 学習開始
                print("学習開始")
                for epoch in range(training_epoch):
                    avg_cost = 0.
                    print(" examples:{}, batch size:{}".format(mnist.train.num_examples, batch_size))
                    total_batch = int(mnist.train.num_examples / batch_size)
                    print(" total_batch:{}".format(total_batch))

                    # ミニバッチ毎ループ
                    for i in range(total_batch):
                        minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)

                        # バッチデータを利用して訓練をフィットさせる
                        sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})

                        # 平均誤差計算
                        avg_cost += sess.run(
                            cost, feed_dict={x: minibatch_x, y: minibatch_y}
                        ) / total_batch

                    # エポック毎にログを出力
                    if epoch % display_step == 0:
                        print("Epoch: {:04d} cost: {:.9f}".format(epoch + 1, avg_cost))
                        accuracy = sess.run(
                            eval_op,
                            feed_dict={x: mnist.validation.images,
                                       y: mnist.validation.labels}
                        )
                        print("Validation Error: {}".format(1 - accuracy))
                        summary_str = sess.run(
                            summary_op,
                            feed_dict={x: minibatch_x, y: minibatch_y}
                        )
                        summary_writer.add_summary(summary_str, sess.run(global_step))
                        saver.save(
                            sess,
                            os.path.join("logistic_logs", "model-checkpoint"),
                            global_step=global_step
                        )
                print("学習終了")

            print("Optimization Finished!")
            accuracy = sess.run(
                eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels}
            )

            print("Test Accuracy: {}".format(accuracy))

            # ここから手書き文字の認識を始める
            # num_data_pathパスのフォルダに数字画像データを保存して起動する
            # データに対する画像処理は未実装
            num_data_path = 'D:/numdata'
            for filename in os.listdir(num_data_path):
                filepath = os.path.join(num_data_path + '/', filename)

                image = Image.open(filepath).convert('L')
                # バックを白に
                image = Image.frombytes('L', image.size,
                                        bytes(map(lambda x: 255 if (x > 160) else x,
                                                  image.getdata())))
                image.thumbnail((28, 28))
                # image.show()

                # input_data の形に揃える
                image = map(lambda x: 255 - x, image.getdata())
                image = np.fromiter(image, dtype=np.uint8)
                image = image.reshape(1, 784)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)

                # 学習データと読み込んだ数値との比較を行う
                p = sess.run(output, feed_dict={x: image, y: [[0.0] * 10]})[0]
                print("File Path:{} --> Result:{}".format(filepath, np.argmax(p)))
                # print(p)
                print()
