from flappybird_env import Environment, FlappyBirdGym
from agent import MultiFuzzyPpoAgent
import argparse
import tensorboard_easy as te
import tf_util
from utils import set_global_seeds
import os
import time
from multiprocessing import Process, Pipe
import numpy as np



def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Must be True or False")


def eval_nfc(agent, env_name, easy=False):
    pipe_gap = 100
    if easy:
        pipe_gap = 150
    env = FlappyBirdGym(pipe_gap=pipe_gap)
    ep_rew = 0

    state = env.reset()
    done = False
    while not done:
        act = agent.rule_act(state)
        state, rew, done, info = env.step(act)
        ep_rew += rew

    return ep_rew


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_update', type=int, default=1200)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--env', type=str, default='flappybird-v0')
    parser.add_argument('--eval', type=str2bool, default=False)
    parser.add_argument('--logdir', type=str, default='./log/')
    parser.add_argument('--agent', type=str, default='nfcmultippo')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--regular', type=float, default=1.0)
    parser.add_argument('--it', type=str, default='3')
    parser.add_argument('--ent_coef', type=float, default=0)
    parser.add_argument('--nsteps', type=int, default=128)
    parser.add_argument('--nworkers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gae_norm', type=str2bool, default=True)
    parser.add_argument('--grad_clip', type=str2bool, default=True)
    parser.add_argument('--pt', type=str, default='sa_parallel_policy')
    parser.add_argument('--hc', type=str2bool, default=False)
    parser.add_argument('--ss', type=str2bool, default=False)
    parser.add_argument('--num_init_v', type=int, default=100)
    parser.add_argument('--num_ri', type=int, default=800)
    parser.add_argument('--save_dir', type=str, default='./model/')
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--plot_nfc', type=str2bool, default=False)
    parser.add_argument('--rule_type', type=str, default='min')
    parser.add_argument('--use_sigmoid', type=str2bool, default=True)
    parser.add_argument('--start_coef', type=float, default=0.7)
    parser.add_argument('--end_coef', type=float, default=0.1)
    parser.add_argument('--delay_step', type=int, default=1)
    args = parser.parse_args()

    # print(args.num_ep)
    # print(args.render)
    # print(args.env)
    # print(args.eval)
    # print(args.logdir)
    # print(args.agent)
    # print(args.seed)
    # print(args.regular)
    # print(args.it)
    # print(args.ent_coef)
    set_global_seeds(args.seed)
    # make sess
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tf_util.make_session(make_default=True)

    NUM_WORKER = args.nworkers
    workers = []
    parent_conns = []
    child_conns = []
    for idx in range(NUM_WORKER):
        parent_conn, child_conn = Pipe()
        worker = Environment(idx, child_conn, seed=args.seed + idx, visualize=False, delay_step=args.delay_step)
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
    # env.seed(args.seed)
    # print('action space:', env.action_space)
    # print('state space:', env.observation_space)
    # print('state space high:', env.observation_space.high)
    # print('state space low:', env.observation_space.low)
    input_size = 5
    # input_size = env.observation_space.shape[0]
    # output_size = env.action_space.n
    output_size = 2

    if args.agent == 'nfcmultippo':
        agent = MultiFuzzyPpoAgent(input_size, output_size, ent_coef=args.ent_coef, nsteps=args.nsteps, num_update=args.num_ri, pt=args.pt, start_coef=args.start_coef, end_coef=args.end_coef, grad_clip=args.grad_clip, gae_norm=args.gae_norm, num_worker=args.nworkers, batch_size=args.batch_size, hc=args.hc )
        logdir = args.logdir + 'mnfc_delay_' + str(args.delay_step) + '_' + str(args.nworkers) + 'workers_' + 'grad_clip_' + str(args.grad_clip) + '_gae_norm_' + str(args.gae_norm) + '_ri'+ str(args.num_ri) + '_' + str(args.start_coef) + '-' + str(args.end_coef) + '_' + args.pt + '_hc_' + str(args.hc) + '_iv' + str(args.num_init_v) + '_' + str(args.seed)
        save_dir = args.save_dir + 'mnfc_delay_' + str(args.delay_step) + '_' + str(args.nworkers) + 'workers_' + 'grad_clip_' + str(args.grad_clip) + '_gae_norm_' + str(args.gae_norm) + '_ri'+ str(args.num_ri) + '_' + str(args.start_coef) + '-' + str(args.end_coef) + '_' + str(args.seed) + '_' + args.pt + '_hc_' + str(args.hc) + '_iv' + str(args.num_init_v)
        print(logdir)
        print('Using nfc ppo agent now!')
    else:
        raise NotImplementedError

    logger = te.Logger(logdir)
    if args.plot_nfc:
        logger_nfc = te.Logger(logdir+'_rule')
    # init v net using rule
    if args.num_init_v > 0:
        print('Initializing V net!')
        state = np.zeros([NUM_WORKER, input_size])
        init_v_count = 0
        while True:
            for i in range(args.nsteps):
                act = agent.act(state)
                for parent_conn, action in zip(parent_conns, act):
                    parent_conn.send(action)
                state, rew, done = [], [], []
                for parent_conn in parent_conns:
                    s, r, d, _ = parent_conn.recv()
                    state.append(s)
                    rew.append(r)
                    done.append(d)
                agent.save_rew_and_done(rew, done)
            agent.get_v_last(state)
            init_v_count += 1
            init_vf_loss = agent.init_v()
            logger.log_scalar('init_V_loss', init_vf_loss, step=init_v_count)
            # agent.clear_buf()
            if init_v_count >= args.num_init_v:
                print('V net has been initialized!')
                break
    # training process
    state = np.zeros([NUM_WORKER, input_size])
    ep_count = 0
    ep_reward = 0
    ep_length = 0
    lasted_ep_r = 0
    update_count = 0
    sample_idx = 0
    while True:
        rollout_start = time.time()
        for i in range(args.nsteps):
            act = agent.act(state)
            for parent_conn, action in zip(parent_conns, act):
                parent_conn.send(action)
            state, rew, done = [], [], []
            for parent_conn in parent_conns:
                s, r, d, _ = parent_conn.recv()
                state.append(s)
                rew.append(r)
                done.append(d)
            agent.save_rew_and_done(rew, done)
            ep_reward += rew[sample_idx]
            ep_length += 1
            if done[sample_idx]:
                ep_count += 1
                logger.log_scalar('ep_r', ep_reward, step=ep_count)
                logger.log_scalar('ep_l', ep_length, step=ep_count)
                print('Episode %d reward %f' % (ep_count, ep_reward))
                lasted_ep_r = ep_reward
                ep_reward = 0
                ep_length = 0
                # reset_start = time.time()
                # state = env.reset()
                # reset_end = time.time()
                # print('Reset time: %s Seconds' % (reset_end - reset_start))
        rollout_end = time.time()
        # print('Rollout time: %s Seconds' % (rollout_end - rollout_start))
        logger.log_scalar('rollout_time', rollout_end - rollout_start, step=update_count + 1)
        # if ep_count >= args.num_ep:
        #     break
        agent.get_v_last(state)
        update_count += 1
        if update_count % args.save_interval == 0:
            tf_util.save_state(save_dir+'/model.ckpt', update_count)
        if args.ss:
            loss, pg_loss, vf_loss, ent_loss, ss_loss, _ = agent.update()
            logger.log_scalar('ss_loss', ss_loss, step=update_count)
        else:
            start = time.time()
            pg_loss, vf_loss, ent_loss = agent.update()
            end = time.time()
            # print('Updating time: %s Seconds'%(end-start))
            logger.log_scalar('update_time', end-start, step=update_count)
        if args.plot_nfc:
            nfc_r = eval_nfc(agent, args.env)
            logger_nfc.log_scalar('update_r', nfc_r, step=update_count)
        logger.log_scalar('update_r', lasted_ep_r, step=update_count)
        # logger.log_scalar('tot', loss, step=update_count)
        logger.log_scalar('pg_loss', pg_loss, step=update_count)
        logger.log_scalar('vf_loss', vf_loss, step=update_count)
        logger.log_scalar('ent_loss', ent_loss, step=update_count)
        agent.clear_buf()
        if update_count >= args.num_update:
            break




if __name__ == '__main__':
    main()