# coding: utf-8
from time import time

from models.Gan import Gan
from models.att_leak.LeakganDataLoader import DataLoader, DisDataloader
from models.att_leak.LeakganDiscriminator import Discriminator, Att_dis
from models.att_leak.LeakganGenerator import Generator
from models.att_leak.LeakganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *
import os

def pre_train_epoch_gen(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, _, _, gen_summary= trainable_model.pretrain_step(sess, batch, .8)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses), gen_summary


def generate_samples_gen(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True, train=0):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, 1.0, train))

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


class Att_leakgan(Gan):
    def __init__(self, pre_epoch_num = 80,dis_epoch = 100,adversarial_epoch_num = 1,reward_co=0.3,step=100,dis_dim=64):
        super().__init__()
        # you can change parameters, generator here
        self.log_dir = 'board/'
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        # flags = tf.app.flags
        # FLAGS = flags.FLAGS
        # flags.DEFINE_boolean('restore', False, 'Training or testing a model')
        # flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
        # flags.DEFINE_integer('length', 20, 'The length of toy data')
        # flags.DEFINE_string('model', "", 'Model NAME')
        # flags.DEFINE_string('f', '', 'kernel')
        # flags.DEFINE_string('g', '', 'kernel')
        # flags.DEFINE_string('t', '', 'kernel')
        # self.sequence_length = FLAGS.length
        self.sequence_length = 20
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 256
        self.start_token = 0
        self.dis_embedding_dim = 64
        self.goal_size = 16
        # 训练调参
        # self.dis_dim=dis_dim
        self.pre_epoch_num = pre_epoch_num
        self.dis_epoch = dis_epoch
        self.adversarial_epoch_num = adversarial_epoch_num
        self.reward_co=reward_co
        self.step=step
        self.step_size=8

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.D_input_x: x_batch,
                self.discriminator.D_input_y: y_batch,
            }
            _, _ = self.sess.run([self.discriminator.D_loss, self.discriminator.D_train_op], feed)
            self.generator.update_feature_function(self.discriminator)

    def evaluate(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()


    def init_oracle_trainng(self, oracle=None):
        goal_out_size = self.emb_dim
        end_token = 4681
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=self.step_size,
                                      l2_reg_lambda=self.l2_reg_lambda)
        # add
        self.set_discriminator(discriminator)
        # reward_co=self.reward_co
        att_model=Att_dis(vocab_size=self.vocab_size, emd_dim=self.emb_dim, sequence_length=self.sequence_length,
                        batch_size=self.batch_size, sess=self.sess,end_token=end_token)
        self.att_model = att_model

        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              att_model=att_model, step_size=self.step_size, sess=self.sess,end_token=end_token)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)



    def train_oracle(self, data_loc=None):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-leakgan.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})


        print('start pre-train generator:')
        self.generator.att_model = self.att_model
        gen_pre_writer = tf.summary.FileWriter(self.log_dir + '/gen_pre', self.sess.graph)
        for epoch in range(self.pre_epoch_num):
            # print('epoch begin')
            start = time()
            # print('pre_train begin')
            loss, gen_summary = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            gen_pre_writer.add_summary(gen_summary, epoch)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                # get_real_test_file()
                self.evaluate()
        gen_pre_writer.close()
        os.rename(self.test_file,'save/pretrain_file.txt')



        print('start pre-train att_model')
        self.reset_epoch()
        att_pre_writer = tf.summary.FileWriter(self.log_dir + '/att_pre', self.sess.graph)
        tmp_count=0
        for epoch in range(self.pre_epoch_num):
            loss_store = []
            start = time()
            for it in range(self.gen_data_loader.num_batch):
                tmp_count+=1
                batch = self.gen_data_loader.next_batch()
                # 会对模型进行参数更新
                _, d_loss, att_summary = self.att_model.train(self.sess, batch)
                att_pre_writer.add_summary(att_summary, tmp_count)

                loss_store.append(d_loss)
                # 这里的pretrain的dataloader是真实文本
            end = time()
            mean_dloss = np.mean(loss_store)
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start)+'\t mean_loss:'+str(mean_dloss))
        att_pre_writer.close()


        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()



        # 这里的feature提取器是没有训练的，效果肯定会比最优要差，
        # 可以把discriminator提前
        # self.generator_file = 'save/generator.txt'
        #         self.test_file = 'save/test_file.txt'


        print('start interval training')
        self.reset_epoch()
        ad_writer = tf.summary.FileWriter(self.log_dir + '/ad', self.sess.graph)
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        tmp_count=0
        for epoch in range(self.adversarial_epoch_num//10):
            for epoch_ in range(10):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1):
                    tmp_count+=1
                    samples = self.generator.generate(self.sess, 1)
                    rewards = self.reward.get_reward(samples)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1
                    }
                    _, _, g_loss, w_loss, ad_summary = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, self.generator.ad_summary], feed_dict=feed)
                    ad_writer.add_summary(ad_summary, tmp_count)

                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                end = time()
                self.add_epoch()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch_ % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                    # get_real_test_file()
                    self.evaluate()


            for epoch_ in range(5):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()
        ad_writer.close()


    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        # 控制台直接运行函数输出(38, 4682)
        # end_token 4681
        # start_token是0，seq中oracle文件是转码后的文本，里面没有start_token，但是运行的时候起始输入是0对应的向量
        # 其实0对应的是个单词，并不是start_token，但是初始化统一为他也行
        # return sequence_len+1, len(word_index_dict) + 1
        self.sequence_length, self.vocab_size = text_precess(data_loc)
        end_token=self.vocab_size-1
        # self.sequence_length += 1
        ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # goal_out_size = sum(self.num_filters)
        goal_out_size = self.emb_dim
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=self.step_size,
                                      l2_reg_lambda=self.l2_reg_lambda)
        # add
        self.set_discriminator(discriminator)
        # reward_co=self.reward_co
        att_model=Att_dis(vocab_size=self.vocab_size, emd_dim=self.emb_dim, sequence_length=self.sequence_length,
                        batch_size=self.batch_size, sess=self.sess,end_token=end_token)
        self.att_model = att_model

        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              att_model=att_model, step_size=self.step_size, sess=self.sess,end_token=end_token)
        self.set_generator(generator)
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length,end_token=end_token)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.sess.run(tf.global_variables_initializer())

        ## !!!!!!!!测试
        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-leakgan-real.csv', 'w')
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)

        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})


        print('start pre-train att_model')
        self.reset_epoch()
        att_pre_writer = tf.summary.FileWriter(self.log_dir + '/att_pre', self.sess.graph)
        tmp_count=0
        for epoch in range(self.pre_epoch_num):
            loss_store = []
            start = time()
            for it in range(self.gen_data_loader.num_batch):
                tmp_count+=1
                batch = self.gen_data_loader.next_batch()
                # 会对模型进行参数更新
                _, d_loss, att_summary = self.att_model.train(self.sess, batch)
                att_pre_writer.add_summary(att_summary, tmp_count)

                loss_store.append(d_loss)
                # 这里的pretrain的dataloader是真实文本
            end = time()
            mean_dloss = np.mean(loss_store)
            self.add_epoch()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start)+'\t mean_loss:'+str(mean_dloss))
        att_pre_writer.close()


        print('start pre-train generator:')
        self.generator.att_model = self.att_model
        gen_pre_writer = tf.summary.FileWriter(self.log_dir + '/gen_pre', self.sess.graph)
        for epoch in range(self.pre_epoch_num):
            # print('epoch begin')
            start = time()
            # print('pre_train begin')
            loss, gen_summary = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            gen_pre_writer.add_summary(gen_summary, epoch)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()
        gen_pre_writer.close()
        os.rename(self.test_file,'save/pretrain_file.txt')

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()



        # 这里的feature提取器是没有训练的，效果肯定会比最优要差，
        # 可以把discriminator提前
        # self.generator_file = 'save/generator.txt'
        #         self.test_file = 'save/test_file.txt'


        print('start interval training')
        self.reset_epoch()
        ad_writer = tf.summary.FileWriter(self.log_dir + '/ad', self.sess.graph)
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        tmp_count=0
        for epoch in range(self.adversarial_epoch_num//10):
            for epoch_ in range(10):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1):
                    tmp_count+=1
                    samples = self.generator.generate(self.sess, 1)
                    rewards = self.reward.get_reward(samples)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1
                    }
                    _, _, g_loss, w_loss, ad_summary = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, self.generator.ad_summary], feed_dict=feed)
                    ad_writer.add_summary(ad_summary, tmp_count)

                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                end = time()
                self.add_epoch()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch_ % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                    get_real_test_file()
                    self.evaluate()


            # for epoch_ in range(5):
            #     start = time()
            #     loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            #     end = time()
            #     print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
            #     if epoch % 5 == 0:
            #         generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
            #                              self.generator_file)
            #         get_real_test_file()
            #         # self.evaluate()

            for epoch_ in range(5):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()
        ad_writer.close()