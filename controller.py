import tensorflow as tf
from baselines.common.distributions import make_pdtype
from gym import spaces
import tf_util
from utils import fc



class Condition(object):
    def __init__(self, mf):
        '''
        :param mf: Membership function
        '''
        self.mf = mf

    def get_value(self, value):
        return self.mf(value)


class FuzzyRule(object):
    def __init__(self, precondiontion_list, act_str, inverse_amf, rule_description=''):
        '''
        :param precondiontion_list:  A list of precondition
        :param inverse_amf: Inverse action membership function
        '''
        self.rule_description = rule_description
        self.preconditions = precondiontion_list
        self.act_str = act_str
        self.pc_num = len(self.preconditions)
        self.iamf = inverse_amf

    def get_result(self, value_list):
        # assert len(value_list) == self.pc_num
        result_value_list = []
        for i in range(self.pc_num):
            result_value_list.append(self.preconditions[i].get_value(value_list[i]))
        pc_result_value = min(result_value_list)
        action_value = self.iamf(pc_result_value)

        # return {self.act_str:(action_value, pc_result_value)}
        return (action_value, pc_result_value)

    def get_values(self, value_list):
        # Used in NeuralFuzzyController
        # return all preconditions' strength
        result_value_list = []
        for i in range(len(value_list)):
            result_value_list.append(1.0)
        for i in range(self.pc_num):
            result_value_list[i] = self.preconditions[i].get_value(value_list[i])

        return result_value_list


class FuzzyLogicController(object):
    def __init__(self, rule_list):
        '''
        :param rule_list:  A list including several FuzzyRules.
        '''
        self.rule_list = rule_list
        self.rule_num = len(self.rule_list)

    def __str__(self):
        str = self.rule_list[0].rule_description
        for r in self.rule_list[1:]:
            str = str + ' ' + r.rule_description
        return str

    def get_action(self, state_dict):
        # state_name = ['TH', 'THV', 'X', 'XV']
        # state = [state_dict[state_name[0]], state_dict[state_name[1]], state_dict[state_name[2]], state_dict[state_name[3]]]
        state_name = ['X', 'XV']
        state = [state_dict[state_name[0]], state_dict[state_name[1]]]
        action_list =[]
        w_sum = 0
        w_z_sum = 0
        for i in range(self.rule_num):
            a = self.rule_list[i].get_result(state)
            action_list.append(a)
            w_sum += a[1]
            w_z_sum += a[0] * a[1]
        return w_z_sum / (w_sum+0.00001) # avoid 0 is divided.


class FbNeuralFuzzyController(object):
    def __init__(self, rule_list, input_size=5, output_size=2):
        '''
        :param rule_list:  A list including several FuzzyRules.
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.rule_list = rule_list
        self.rule_num = len(self.rule_list)
        with tf.variable_scope('nfc'):
            self.ph_list = []
            for i in range(len(self.rule_list)):
                self.ph_list.append(tf.placeholder(tf.float32, shape=[None, input_size], name='ph_r' + str(i)))

            self.logits = self._build_net()
            self.pdtype = make_pdtype(spaces.Discrete(output_size))
            self.ph_act = self.pdtype.sample_placeholder([None], name='ph_nfc_act')
            self.pd = self.pdtype.pdfromflat(self.logits*100)
            self.a_samp = self.pd.sample()
            self.a_mode = self.pd.mode()
        allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nfc')
        tf_util.display_var_info(allvars)
        tf.get_default_session().run(tf.global_variables_initializer())

    def _build_net(self):

        self.w1_list = []
        self.b1_list = []
        for i in range(len(self.rule_list)):
            self.w1_list.append(tf.Variable(tf.ones([self.input_size]), dtype=tf.float32, name='w1_r' + str(i)))
            self.b1_list.append(tf.Variable(tf.ones([self.input_size])*0.1, dtype=tf.float32, name='b1_r' + str(i)))
        r_strength_list = []
        for i in range(len(self.rule_list)):
            r_strength_list.append(tf.reduce_min((tf.multiply(self.ph_list[i], self.w1_list[i]) + self.b1_list[i]), axis=1))
        self.w2_list = []
        for i in range(len(self.rule_list)):
            self.w2_list.append(tf.Variable(initial_value=1.0, dtype=tf.float32, name='w2_r' + str(i)))
        fix_r_strength_list = []
        for i in range(len(self.rule_list)):
            fix_r_strength_list.append(tf.multiply(r_strength_list[i], self.w2_list[i]))
        # TODO adaptive to Flappy Bird only
        n_strength_tensor1 = fix_r_strength_list[1]
        n_strength_tensor2 = fix_r_strength_list[3]
        t_strength_tensor1 = fix_r_strength_list[0]
        t_strength_tensor2 = fix_r_strength_list[2]
        self.n1 = fix_r_strength_list[1]
        self.n2 = fix_r_strength_list[3]
        self.t1 = fix_r_strength_list[0]
        self.t2 = fix_r_strength_list[2]
        n_strength_tensor = tf.reduce_mean(tf.stack([n_strength_tensor1, n_strength_tensor2], axis=0), axis=0)
        t_strength_tensor = tf.reduce_mean(tf.stack([t_strength_tensor1, t_strength_tensor2], axis=0), axis=0)
        logits = tf.stack([n_strength_tensor,t_strength_tensor], axis=0)
        logits = tf.transpose(logits)

        return logits


    def get_feed_dict(self, state_dict_list):
        feed_dict ={
            self.ph_list[0]:[],
            self.ph_list[1]:[],
            self.ph_list[2]:[],
            self.ph_list[3]:[],
        }
        state_name = ['Y', 'VY', 'D', 'YTY', 'YBY']
        for i in range(len(state_dict_list)):
            state = [state_dict_list[i][state_name[0]], state_dict_list[i][state_name[1]], state_dict_list[i][state_name[2]],
                     state_dict_list[i][state_name[3]], state_dict_list[i][state_name[4]]]
            for j in range(self.rule_num):
                feed_dict[self.ph_list[j]].append(self.rule_list[j].get_values(state))

        return feed_dict

    def __str__(self):
        str = self.rule_list[0].rule_description
        for r in self.rule_list[1:]:
            str = str + ' ' + r.rule_description
        return str

    def call(self, state_dict):
        feed_dict = self.get_feed_dict([state_dict])
        a, l= tf.get_default_session().run([self.a_samp, self.logits], feed_dict=feed_dict)
        # if a[0] == 1:
        #     print(state_dict)
        #     n1, n2, t1, t2= tf.get_default_session().run([self.n1, self.n2, self.t1, self.t2], feed_dict=feed_dict)
        #     print(n1, n2, t1, t2)
        return a[0]

    def display_w(self):
        w = tf.get_default_session().run([self.w1_list, self.w2_list, self.b1_list])
        print(w)


def get_rule_from_str(str, mf_dict, imf_dict):
    state_name = ['Y', 'VY', 'D', 'YTY', 'YBY']
    # state_name = ['X', 'Y', 'VX', 'VY', 'TH', 'THV']
    # state_name = ['TH', 'THV', 'X', 'XV']
    pc_str = str.split('=>')[0]
    act_str = str.split('=>')[1]
    pc = []
    for i, c_str in enumerate(pc_str.split(' ')):
        pc.append(Condition(mf_dict[state_name[i] + c_str]))
    return FuzzyRule(pc, act_str, imf_dict['I'+'F'+act_str], rule_description=str)


def get_controller_from_rules(rs, mf_dict, imf_dict):
    rule_list = []
    for rule in rs:
        fr = get_rule_from_str(rule, mf_dict, imf_dict)
        rule_list.append(fr)
    return FuzzyLogicController(rule_list)


def get_neural_controller_from_rules(rs, mf_dict, imf_dict):
    rule_list = []
    print(rs)
    for rule in rs:
        fr = get_rule_from_str(rule, mf_dict, imf_dict)
        rule_list.append(fr)
    return FbNeuralFuzzyController(rule_list)




def get_rule_list(rs, mf_dict, imf_dict):
    rule_list = []
    for rule in rs:
        fr = get_rule_from_str(rule, mf_dict, imf_dict)
        rule_list.append(fr)
    return rule_list


class LearnFuzzyController(object):
    def __init__(self, input_size=4, output_size=2, rule_type='min', sigmoid=True):

        self.input_size = input_size
        self.output_size = output_size
        self.rule_type = rule_type
        self.use_sigmoid = sigmoid
        # TODO
        # self.mf_dict = {0:[],
        #                 1:[],
        #                 2:[],
        #                 3:[]}
        self.mf_dict = {0:[],
                        1:[]}
        with tf.variable_scope('lfc'):
            self.ph_ob_list =[]
            for i in range(input_size):
                self.ph_ob_list.append(tf.placeholder(tf.float32, shape=[None, 1], name='ph_ob' + str(i)))
            self.ph_ob = tf.placeholder(tf.float32, shape=[None, input_size], name='ph_ob')
            self.logits = self._build_net()
            self.critic_out = self._build_critic(self.ph_ob)
            self.pdtype = make_pdtype(spaces.Discrete(output_size))
            self.ph_act = self.pdtype.sample_placeholder([None], name='ph_lfc_act')
            if self.use_sigmoid:
                self.pd = self.pdtype.pdfromflat(self.logits * 5) # sharpen the distribution
            else:
                self.pd = self.pdtype.pdfromflat(self.logits)
            self.a_samp = self.pd.sample()
            self.a_mode = self.pd.mode()
            self.nlp_samp = self.pd.neglogp(self.a_samp)

    def _build_net(self):

        r0 = self._build_learn_rule(number=0)
        r1 = self._build_learn_rule(number=1)
        logits = tf.stack([r0, r1], axis=1)
        return logits

    # def _build_net(self):
    #
    #     a0_r0 = self._build_learn_rule(number=0)
    #     a0_r1 = self._build_learn_rule(number=1)
    #     a0_s = tf.stack([a0_r0, a0_r1], axis=1)
    #     a0_s = tf.reduce_mean(a0_s, axis=1)
    #     a1_r0 = self._build_learn_rule(number=2)
    #     a1_r1 = self._build_learn_rule(number=3)
    #     a1_s = tf.stack([a1_r0, a1_r1], axis=1)
    #     a1_s = tf.reduce_mean(a1_s, axis=1)
    #     logits = tf.stack([a0_s, a1_s], axis=1)
    #     return logits

    def _build_learn_rule(self, number):
        with tf.variable_scope('lrule' + str(number)):
            lmf_list = []
            for i in range(self.input_size):
                mf = self._build_learn_mf(self.ph_ob_list[i], number=i)
                lmf_list.append(mf)
                self.mf_dict[number].append(mf)
            precondition_strengths = tf.stack(lmf_list, axis=1)  # (N, input_size)
            if self.rule_type == 'min':
                rule_strength = tf.reduce_min(precondition_strengths, axis=1)  # (N,)
            elif self.rule_type == 'mean':
                rule_strength = tf.reduce_mean(precondition_strengths, axis=1)  # (N,)
            return rule_strength

    # def _build_learn_mean_rule(self, number):
    #     with tf.variable_scope('lrule' + str(number)):
    #         lmf_list = []
    #         for i in range(self.input_size):
    #             mf = self._build_learn_mf(self.ph_ob_list[i], number=i)
    #             lmf_list.append(mf)
    #             self.mf_dict[number].append(mf)
    #         precondition_strengths = tf.stack(lmf_list, axis=1) # (N, input_size)
    #         rule_strength = tf.reduce_mean(precondition_strengths, axis=1) # (N,)
    #         return rule_strength

    def _build_learn_mf(self, ph, number):
        with tf.variable_scope('mf' + str(number)):
            activ = tf.nn.relu
            x = activ(fc(ph, 'fc1', nh=10, init_scale=0.1))
            x = activ(fc(x, 'fc2', nh=10, init_scale=0.1))
            out = fc(x, 'out', nh=1, init_scale=0.01) # (N, 1)
            if self.use_sigmoid:
                out = tf.nn.sigmoid(out) # shape = (N, 1) , map to (0, 1)
            out = tf.reshape(out,shape=[-1]) # (N,)
        return out

    def _build_critic(self, ph, scope='critic'):
        with tf.variable_scope(scope):
            activ = tf.nn.relu
            x = activ(fc(ph, 'fc1', nh=32, init_scale=0.1))
            x = activ(fc(x, 'fc2', nh=32, init_scale=0.1))
            out = fc(x, 'out', nh=1, init_scale=0.01)
        return out

    def get_feed_dict(self, state_dict_list):
        feed_dict ={
            self.ph_ob_list[0]:[],
            self.ph_ob_list[1]:[],
            self.ph_ob_list[2]:[],
            self.ph_ob_list[3]:[],
            self.ph_ob:[]
        }
        state_name = ['TH', 'THV', 'X', 'XV']
        for i in range(len(state_dict_list)):
            state = [state_dict_list[i][state_name[0]], state_dict_list[i][state_name[1]],
                     state_dict_list[i][state_name[2]],state_dict_list[i][state_name[3]]]
            for j in range(self.input_size):
                feed_dict[self.ph_ob_list[j]].append([state[j]])
            feed_dict[self.ph_ob].append(state)

        return feed_dict

    def call(self, state_dict):
        feed_dict = self.get_feed_dict([state_dict])
        a, v, nlp = tf.get_default_session().run([self.a_samp, self.critic_out, self.nlp_samp], feed_dict=feed_dict)

        return a[0], v[0][0], nlp[0]

    def mf_call(self, x, rule_number, mf_number):
        feed_dict = {
            self.ph_ob_list[mf_number]: [[x]]
        }
        s = tf.get_default_session().run(self.mf_dict[rule_number][mf_number], feed_dict=feed_dict)

        return s[0]


class MixFuzzyController(object):
    def __init__(self, rule_list, input_size=4, output_size=2, rule_type='min', sigmoid=True):
        self.rule_list = rule_list
        self.input_size = input_size
        self.output_size = output_size
        self.rule_type = rule_type
        self.use_sigmoid = sigmoid
        # TODO
        # self.mf_dict = {0:[],
        #                 1:[],
        #                 2:[],
        #                 3:[]}
        self.mf_dict = {0:[],
                        1:[],
                        2:[]}
        with tf.variable_scope('mfc'):
            self.ph_ob_list =[]
            for i in range(input_size):
                self.ph_ob_list.append(tf.placeholder(tf.float32, shape=[None, 1], name='ph_ob' + str(i)))
            self.fr_ph_list = []
            for i in range(len(self.rule_list)):
                self.fr_ph_list.append(tf.placeholder(tf.float32, shape=[None, input_size], name='fr_ph_r' + str(i)))
            self.ph_ob = tf.placeholder(tf.float32, shape=[None, input_size], name='ph_ob')
            self.fr_w1_list = []
            self.fr_b1_list = []
            self.logits = self._build_net()
            self.critic_out = self._build_critic(self.ph_ob)
            self.pdtype = make_pdtype(spaces.Discrete(output_size))
            self.ph_act = self.pdtype.sample_placeholder([None], name='ph_lfc_act')
            if self.use_sigmoid:
                self.pd = self.pdtype.pdfromflat(self.logits * 5) # sharpen the distribution
            else:
                self.pd = self.pdtype.pdfromflat(self.logits)
            self.a_samp = self.pd.sample()
            self.a_mode = self.pd.mode()
            self.nlp_samp = self.pd.neglogp(self.a_samp)

    def _build_net(self):

        r0 = self._build_learn_rule(number=0)
        r1 = self._build_learn_rule(number=1)
        r2 = self._build_learn_rule(number=2)

        fr0 = self._build_fix_rule(number=0)
        fr2 = self._build_fix_rule(number=1)

        a0_s = tf.stack([r0, tf.stop_gradient(fr0)], axis=1)
        a0_s = tf.reduce_mean(a0_s, axis=1)

        a1_s = r1

        a2_s = tf.stack([r2, tf.stop_gradient(fr2)], axis=1)
        a2_s = tf.reduce_mean(a2_s, axis=1)

        logits = tf.stack([a0_s, a1_s, a2_s], axis=1)
        return logits

    # def _build_net(self):
    #
    #     a0_r0 = self._build_learn_rule(number=0)
    #     a0_r1 = self._build_learn_rule(number=1)
    #     a0_s = tf.stack([a0_r0, a0_r1], axis=1)
    #     a0_s = tf.reduce_mean(a0_s, axis=1)
    #     a1_r0 = self._build_learn_rule(number=2)
    #     a1_r1 = self._build_learn_rule(number=3)
    #     a1_s = tf.stack([a1_r0, a1_r1], axis=1)
    #     a1_s = tf.reduce_mean(a1_s, axis=1)
    #     logits = tf.stack([a0_s, a1_s], axis=1)
    #     return logits

    def _build_learn_rule(self, number):
        with tf.variable_scope('lrule' + str(number)):
            lmf_list = []
            for i in range(self.input_size):
                mf = self._build_learn_mf(self.ph_ob_list[i], number=i)
                lmf_list.append(mf)
                self.mf_dict[number].append(mf)
            precondition_strengths = tf.stack(lmf_list, axis=1) # (N, input_size)
            if self.rule_type == 'min':
                rule_strength = tf.reduce_min(precondition_strengths, axis=1) # (N,)
            elif self.rule_type == 'mean':
                rule_strength = tf.reduce_mean(precondition_strengths, axis=1)  # (N,)
            return rule_strength

    # def _build_learn_mean_rule(self, number):
    #     with tf.variable_scope('lrule' + str(number)):
    #         lmf_list = []
    #         for i in range(self.input_size):
    #             mf = self._build_learn_mf(self.ph_ob_list[i], number=i)
    #             lmf_list.append(mf)
    #             self.mf_dict[number].append(mf)
    #         precondition_strengths = tf.stack(lmf_list, axis=1) # (N, input_size)
    #         rule_strength = tf.reduce_mean(precondition_strengths, axis=1) # (N,)
    #         return rule_strength

    def _build_fix_rule(self, number):
        with tf.variable_scope('frule' + str(number)):
            self.fr_w1_list.append(tf.Variable(tf.ones([self.input_size]), dtype=tf.float32, name='w1_r' + str(number)))
            self.fr_b1_list.append(tf.Variable(tf.zeros([self.input_size]), dtype=tf.float32, name='b1_r' + str(number)))
            rule_strength = tf.reduce_min((tf.multiply(self.fr_ph_list[number], self.fr_w1_list[number]) + self.fr_b1_list[number]), axis=1)

        return rule_strength

    def _build_learn_mf(self, ph, number):
        with tf.variable_scope('mf' + str(number)):
            activ = tf.nn.relu
            x = activ(fc(ph, 'fc1', nh=10, init_scale=0.1))
            x = activ(fc(x, 'fc2', nh=10, init_scale=0.1))
            out = fc(x, 'out', nh=1, init_scale=0.01) # (N, 1)
            if self.use_sigmoid:
                out = tf.nn.sigmoid(out) # shape = (N, 1) , map to (0, 1)
            out = tf.reshape(out,shape=[-1]) # (N,)
        return out

    def _build_critic(self, ph, scope='critic'):
        with tf.variable_scope(scope):
            activ = tf.nn.relu
            x = activ(fc(ph, 'fc1', nh=32, init_scale=0.1))
            x = activ(fc(x, 'fc2', nh=32, init_scale=0.1))
            out = fc(x, 'out', nh=1, init_scale=0.01)
        return out

    def get_feed_dict(self, state_dict_list):
        feed_dict ={
            self.ph_ob_list[0]: [],
            self.ph_ob_list[1]: [],

            self.fr_ph_list[0]: [],
            self.fr_ph_list[1]: [],

            self.ph_ob: []
        }
        state_name = ['X', 'XV']
        for i in range(len(state_dict_list)):
            state = [state_dict_list[i][state_name[0]], state_dict_list[i][state_name[1]]]
            for j in range(self.input_size):
                feed_dict[self.ph_ob_list[j]].append([state[j]])
            for j in range(len(self.rule_list)):
                feed_dict[self.fr_ph_list[j]].append(self.rule_list[j].get_values(state))
            feed_dict[self.ph_ob].append(state)

        return feed_dict


    def call(self, state_dict):
        feed_dict = self.get_feed_dict([state_dict])
        a, v, nlp, am, l = tf.get_default_session().run([self.a_samp, self.critic_out, self.nlp_samp, self.a_mode, self.logits], feed_dict=feed_dict)
        print(a, am, l)
        return a[0], v[0][0], nlp[0]

    def mf_call(self, x, rule_number, mf_number):
        feed_dict = {
            self.ph_ob_list[mf_number]: [[x]]
        }
        s = tf.get_default_session().run(self.mf_dict[rule_number][mf_number], feed_dict=feed_dict)

        return s[0]