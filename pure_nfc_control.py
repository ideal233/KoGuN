from controller import get_neural_controller_from_rules
from fb_rules import rules, membership_functions, imfs
import argparse
import tf_util
from flappybird_env import FlappyBirdGym

def to_state_dict(state):
    return {
        'Y':state[0],
        'VY':state[1],
        'D':state[2],
        'YTY':state[3],
        'YBY':state[4],
    }


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ep', type=int, default=10000)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--env', type=str, default='FlappyBird-v0')
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--logdir', type=str, default='./log/')
    args = parser.parse_args(args=[])
    tf_util.make_session(make_default=True)
    nfc = get_neural_controller_from_rules(rules, membership_functions, imfs)
    env = FlappyBirdGym(display=False, delay_step=1)
    tr = 0
    for i in range(1000):
        state = env.reset()
        # print(state)
        done = False
        er = 0
        while not done:
            # print(state)
            act = nfc.call(to_state_dict(state))
            # time.sleep(0.1)
            # act0 = flc.get_action(to_state_dict(state))
            # print(act)

            state, rew, done, info = env.step(act)
            # if rew > 40:
            #     print(rew)
            er += rew
            # print(state)
            # env.render()
        print(er)
        tr += er
    print(tr/1000.0)




if __name__ == '__main__':
    main()