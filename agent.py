from policy import MultiFuzzyActorCritic
import tensorflow as tf
import tf_util
import numpy as np
import copy


def to_state_dict(state):
    return {
        'Y':state[0],
        'VY':state[1],
        'D':state[2],
        'YTY':state[3],
        'YBY':state[4],
    }


class MultiFuzzyPpoAgent(object):
    def __init__(self, input_size=5, output_size=2, gamma=0.99, lr=1e-4, lamda=0.95, cliprange=0.2, ent_coef=None, nsteps=128, num_worker=1, epoch=3, batch_size=32, grad_clip=True, gae_norm=True, pt='res_policy', num_update=1000, start_coef=0.7, end_coef=0.4, hc=False):
        self.update_count = 0
        self.epoch = epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.lamda = lamda
        self.gae_norm = gae_norm
        self.grad_clip = grad_clip
        self.grad_clip_min = -1.0
        self.grad_clip_max = 1.0
        self.num_worker = num_worker
        self.input_size = input_size
        self.output_size = output_size
        alr = lr
        clr = 5*lr
        self.pt = pt
        self.start_coef = start_coef
        self.end_coef = end_coef
        with tf.variable_scope('agent'):
            self.ph_ret = tf.placeholder(tf.float32, shape=[None,], name='ph_return')
            self.ph_oldnlp = tf.placeholder(tf.float32, shape=[None,], name='ph_neglogp')
            self.ph_adv = tf.placeholder(tf.float32, shape=[None,], name='ph_adv')
            self.pol = MultiFuzzyActorCritic(input_size, output_size, pt=pt, hc=hc)
            nlogp = self.pol.pd.neglogp(self.pol.ph_act)
            ratio = tf.exp(self.ph_oldnlp - nlogp) # - oldlogp - (- logp) = logp - oldlogp = log(p/oldp)
            negadv = - self.ph_adv
            ppo_loss1 = negadv * ratio
            ppo_loss2 = negadv * tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)
            self.pg_loss = tf.reduce_mean(tf.maximum(ppo_loss1, ppo_loss2))
            self.critic_loss = tf.losses.mean_squared_error(self.pol.critic_out, self.ph_ret)

            if ent_coef is not None:
                entropy = tf.reduce_mean(self.pol.pd.entropy())
                self.ent_loss = (- ent_coef) * entropy
                actor_loss = self.pg_loss + self.ent_loss
            else:
                actor_loss = self.pg_loss
            actor_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'agent/fac/'+self.pt)
            critic_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'agent/fac/critic')

            self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=alr)
            actor_grad = self.actor_optimizer.compute_gradients(actor_loss, actor_var)
            if self.grad_clip:
                actor_grad = [(tf.clip_by_value(grad, self.grad_clip_min, self.grad_clip_max), var) for grad, var in
                              actor_grad]
            self.train_a_op = self.actor_optimizer.apply_gradients(actor_grad)

            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=clr)
            critic_grad = self.critic_optimizer.compute_gradients(self.critic_loss, critic_var)
            if self.grad_clip:
                critic_grad = [(tf.clip_by_value(grad, self.grad_clip_min, self.grad_clip_max), var) for grad, var in
                               critic_grad]
            self.train_c_op = self.critic_optimizer.apply_gradients(critic_grad)

            self.state_buf = []
            self.state_dict_buf = []
            self.act_buf = []
            self.v_buf = []
            self.rew_buf = []
            self.nlp_buf = []
            self.done_buf = []
            self.v_last = None
            self.nsteps = nsteps
        allvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent')
        tf_util.display_var_info(allvars)
        tf.get_default_session().run(tf.global_variables_initializer())
        self.update_count = 0
        self.init_count = 0
        self.total_update = num_update

    def get_coef(self):
        start_coef = self.start_coef
        end_coef = self.end_coef
        decay = (start_coef - end_coef) * self.update_count / self.total_update
        coef = start_coef - decay

        return max(coef, end_coef)

    def rule_act(self, state):
        a = self.pol.nfc.call(to_state_dict(state))
        return a

    def act(self, states):
        self.state_buf.append(states)
        self.state_dict_buf.append([to_state_dict(s) for s in states])
        a, v ,nlp = self.pol.call(states, self.get_coef())

        self.act_buf.append(a)
        self.v_buf.append(v)
        self.nlp_buf.append(nlp)

        return a

    def get_v_last(self, nextstates):
        _, v, _ = self.pol.call(nextstates, self.get_coef())
        self.v_last = v

    def eval_act(self, state):
        a = self.pol.eval_call([state])
        return a

    def save_rew_and_done(self, rew, done):
        self.rew_buf.append(rew)
        self.done_buf.append(done)

    def get_returns_and_advs(self):
        assert self.v_last is not None
        ret = []
        adv = []
        # Using GAE here.
        lastgaelam = 0
        for t in range(self.nsteps-1, -1, -1):
            nextvals = self.v_buf[t + 1] if t + 1 < self.nsteps else self.v_last
            delta = self.rew_buf[t] + self.gamma * nextvals * (1 - self.done_buf[t]) - self.v_buf[t]
            lastgaelam = delta + self.gamma * self.lamda * (1 - self.done_buf[t]) * lastgaelam
            adv.append(lastgaelam)
            ret.append(lastgaelam + self.v_buf[t])
        ret.reverse()
        adv.reverse()
        return ret, adv

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        # print(rewards, dones, next_values, values)
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        # print(deltas)
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes, target

    def update(self):
        # rets, advs = self.get_returns_and_advs()
        # states_dict = self.pol.nfc.get_feed_dict(self.state_dict_buf)
        state_dicts = []
        for sds in self.state_dict_buf:
            for sd in sds:
                state_dicts.append(sd)
        rets, advs = [], []
        states = np.array(self.state_buf).transpose([1, 0, 2]).reshape([-1, self.input_size])
        acts = np.array(self.act_buf).transpose().reshape([-1])
        nlps = np.array(self.nlp_buf).transpose().reshape([-1])
        vs = np.array(self.v_buf).transpose().reshape([-1])
        rew_buf = np.array(self.rew_buf).transpose().reshape([-1])
        done_buf = np.array(self.done_buf).astype(np.float).reshape([-1])
        nv_buf = copy.deepcopy(self.v_buf)
        nv_buf.append(self.v_last)
        nv_buf = nv_buf[1:]
        nvs = np.array(nv_buf).transpose().reshape([-1])

        for idx in range(self.num_worker):
            num_step = self.nsteps
            a, t = self.get_gaes(rew_buf[idx * num_step:(idx + 1) * num_step],
                                 done_buf[idx * num_step:(idx + 1) * num_step],
                                 vs[idx * num_step:(idx + 1) * num_step],
                                 nvs[idx * num_step:(idx + 1) * num_step])
            rets.append(t)
            advs.append(a)
        rets = np.hstack(rets)
        advs = np.hstack(advs)
        sample_range = np.arange(len(states))

        for epc in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(states) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                state_batch = [states[i] for i in sample_idx]
                state_dict_batch = [state_dicts[i] for i in sample_idx]
                act_batch = [acts[i] for i in sample_idx]
                adv_batch = [advs[i] for i in sample_idx]
                ret_batch = [rets[i] for i in sample_idx]
                nlp_batch = [nlps[i] for i in sample_idx]
                feed = self.pol.nfc.get_feed_dict(state_dict_batch)
                vfloss, _ = tf.get_default_session().run([self.critic_loss, self.train_c_op],
                                                         feed_dict={
                                                             self.pol.ph_ob: state_batch,
                                                             self.ph_ret: ret_batch
                                                         })
                feed.update({
                    self.pol.ph_coef: [self.get_coef()],
                    self.pol.ph_ob: state_batch,
                    self.pol.ph_act: act_batch,
                    self.ph_adv: adv_batch,
                    self.ph_oldnlp: nlp_batch
                })
                pgloss, ent_loss, _ = tf.get_default_session().run([self.pg_loss, self.ent_loss, self.train_a_op],
                                                                   feed_dict=feed)

        self.clear_buf()
        self.update_count += 1
        return pgloss, vfloss, ent_loss

    def init_v(self):
        # rets, advs = self.get_returns_and_advs()
        rets = []
        states = np.array(self.state_buf).transpose([1, 0, 2]).reshape([-1, self.input_size])
        vs = np.array(self.v_buf).transpose().reshape([-1])
        rew_buf = np.array(self.rew_buf).transpose().reshape([-1])
        done_buf = np.array(self.done_buf).astype(np.float).reshape([-1])
        nv_buf = copy.deepcopy(self.v_buf)
        nv_buf.append(self.v_last)
        nv_buf = nv_buf[1:]
        nvs = np.array(nv_buf).transpose().reshape([-1])

        for idx in range(self.num_worker):
            num_step = self.nsteps
            _, t = self.get_gaes(rew_buf[idx * num_step:(idx + 1) * num_step],
                                 done_buf[idx * num_step:(idx + 1) * num_step],
                                 vs[idx * num_step:(idx + 1) * num_step],
                                 nvs[idx * num_step:(idx + 1) * num_step])
            rets.append(t)

        rets = np.hstack(rets)
        sample_range = np.arange(len(states))

        for epc in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(states) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                state_batch = [states[i] for i in sample_idx]
                ret_batch = [rets[i] for i in sample_idx]
                init_vf_loss, _ = tf.get_default_session().run([self.critic_loss, self.train_c_op],
                                                         feed_dict={
                                                             self.pol.ph_ob: state_batch,
                                                             self.ph_ret: ret_batch
                                                         })

        self.clear_buf()
        self.init_count += 1
        return init_vf_loss

    def clear_buf(self):
        # print(self.rew_buf)
        self.state_buf = []
        self.state_dict_buf = []
        self.act_buf = []
        self.v_buf = []
        self.rew_buf = []
        self.nlp_buf = []
        self.done_buf = []
        self.v_last = None
