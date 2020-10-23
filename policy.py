import tensorflow as tf
from utils import fc
from baselines.common.distributions import make_pdtype
from gym import spaces
from controller import get_neural_controller_from_rules
from fb_rules import imfs, membership_functions, rules
import time


def to_state_dict(state):
    return {
        'Y':state[0],
        'VY':state[1],
        'D':state[2],
        'YTY':state[3],
        'YBY':state[4],
    }


class MultiFuzzyActorCritic(object):
    def __init__(self, input_size=5, output_size=2, pt='sa_parallel_policy', hc=False):
        with tf.variable_scope('fac'):
            self.output_size = output_size
            self.input_size = input_size
            self.ph_ob = tf.placeholder(tf.float32,shape=[None, input_size], name='ph_obs')
            self.pdtype = make_pdtype(spaces.Discrete(output_size))
            self.ph_act = self.pdtype.sample_placeholder([None], name='ph_act')
            self.ph_coef = tf.placeholder(tf.float32,shape=(1,), name='ph_coef')
            self.nfc = get_neural_controller_from_rules(rules, membership_functions, imfs)
            if pt=='sa_parallel_policy':
                self.actor_out, self.fix_policy_pd = self._build_sa_parallel_policy(self.ph_ob)
            elif pt=='hyper_policy':
                self.actor_out, self.fix_policy_pd = self._build_hyper_policy(self.ph_ob)
            else:
                raise NotImplementedError
            self.critic_out = tf.squeeze(self._build_critic(self.ph_ob))
            # self.critic_out = self._build_critic(self.ph_ob)
            self.pd = self.pdtype.pdfromflat(self.actor_out)
            self.a_samp = self.pd.sample()
            self.a_mode = self.pd.mode()
            self.nlp_samp = self.pd.neglogp(self.a_samp)

    def _build_sa_parallel_policy(self, ph, scope='sa_parallel_policy'):
        with tf.variable_scope(scope):
            activ = tf.nn.relu
            sa = tf.concat([ph, tf.stop_gradient(self.nfc.logits)], axis=1)
            h1 = activ(fc(sa, 'w1', nh=32, init_scale=0.1))
            h2 = activ(fc(h1, 'w2', nh=32, init_scale=0.1))
            out = fc(h2, 'w3', nh=self.output_size, init_scale=0.01)
            fix_policy_pd = tf.nn.softmax(out)
            actor_out = ((1 - self.ph_coef) * tf.nn.sigmoid(out) + self.ph_coef * self.nfc.logits) * 10

        return actor_out, fix_policy_pd

    def _build_hyper_policy(self, ph, scope='hyper_policy'):
        with tf.variable_scope(scope):
            activ = tf.nn.relu

            x1 = activ(fc(ph, 'w1_fc1', nh=32, init_scale=0.1))
            x1 = activ(fc(x1, 'w1_fc2', nh=32, init_scale=0.1))
            w1 = fc(x1, 'w1_fc3', nh=32 * self.output_size, init_scale=0.01)
            w1 = tf.reshape(w1, [-1, self.output_size, 32])

            x2 = activ(fc(ph, 'w2_fc1', nh=32, init_scale=0.1))
            x2 = activ(fc(x2, 'w2_fc2', nh=32, init_scale=0.1))
            w2 = fc(x2, 'w2_fc3', nh=32 * self.output_size, init_scale=0.01)
            w2 = tf.reshape(w2, [-1, 32, self.output_size])

            x3 = activ(fc(ph, 'b1_fc1', nh=32, init_scale=0.1))
            x3 = activ(fc(x3, 'b1_fc2', nh=32, init_scale=0.1))
            b1 = fc(x3, 'b1_fc3', nh=32, init_scale=0.01)
            b1 = tf.reshape(b1, [-1, 1, 32])

            p = tf.reshape(tf.stop_gradient(self.nfc.logits), [-1, 1, self.output_size])
            h1 = activ(tf.matmul(p, w1) + b1)
            out = tf.matmul(h1, w2)
            out = tf.reshape(out, [-1, self.output_size])
            fix_policy_pd = tf.nn.softmax(out)
            actor_out = ((1 - self.ph_coef) * tf.nn.sigmoid(out) + self.ph_coef * self.nfc.logits) * 10

        return actor_out, fix_policy_pd

    def _build_critic(self, ph, scope='critic'):
        with tf.variable_scope(scope):
            activ = tf.nn.relu
            x = activ(fc(ph, 'fc1', nh=256, init_scale=0.1))
            x = activ(fc(x, 'fc2', nh=256, init_scale=0.1))
            out = fc(x, 'out', nh=1, init_scale=0.01)
            print(out)
        return out

    def call(self, states, coef):
        feed_dict = self.nfc.get_feed_dict([to_state_dict(s) for s in states])
        feed_dict[self.ph_ob] = states
        feed_dict[self.ph_coef] = [coef]
        ###########
        # feed_dict[self.ph_l] = [[[0.5,0.5]]]
        ###########
        a, v, nlp= tf.get_default_session().run([self.a_samp, self.critic_out, self.nlp_samp], feed_dict=feed_dict)

        return a, v, nlp

    def eval_call(self, state):
        feed = {self.ph_ob: state}
        a = tf.get_default_session().run(self.a_mode, feed_dict=feed)

        return a[0]




