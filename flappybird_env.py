from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from multiprocessing import Pipe, Process

class FlappyBirdGym(object):
    def __init__(self, seed=24, max_episode_length=2000, display=False, pipe_gap=100, skip_frame=0, delay_step=1):
        game = FlappyBird(pipe_gap=pipe_gap)
        self.p = PLE(game, fps=30, display_screen=display, rng=seed)
        self.max_l = max_episode_length
        assert skip_frame == 0
        assert delay_step != 0, 'Delay step must greater than 0'
        self.skip_frame = skip_frame
        self.step_count = 0
        self.delay_step = delay_step
        self.pass_pipe = 0

    def to_state(self, game_state_dict):
        state_list = [i for i in game_state_dict.values()]
        state_list = state_list[:5]
        state_list[3] = state_list[0] - state_list[3]
        state_list[4] = state_list[0] - state_list[4]
        return np.array(state_list)

    def reset(self):
        self.p.init()
        self.p.previous_score = 0
        self.step_count = 0
        self.pass_pipe = 0
        return self.to_state(self.p.getGameState())

    def step(self, act):
        assert act in [0 ,1]
        if act == 0:
            act = None
        else:
            act = 119
        r = self.p.act(act)
        done = self.p.game_over()
        snext = self.to_state(self.p.getGameState())
        # print(self.p.getGameState())
        if r == 1:
            self.pass_pipe += 1
        # FIXME delay_reward and skip_frame have conflict.
        # for i in range(self.skip_frame):
        #     if done:
        #         break
        #     r += self.p.act(None)
        #     done = self.p.game_over()
        #     snext = self.to_state(self.p.getGameState())
            # print(self.p.getGameState())
        if self.pass_pipe == self.delay_step:
            reward = self.delay_step
            self.pass_pipe = 0
        else:
            reward = 0
        if done:
            reward += -5
        info = {}
        self.step_count += 1
        if self.step_count >= self.max_l:
            done = True
        return snext, reward, done, info

class Environment(Process):
    def __init__(self, env_idx, child_conn, seed, pipe_gap=100,visualize=False, delay_step=0):
        super(Environment, self).__init__()
        self.env = FlappyBirdGym(display=visualize, seed=seed, pipe_gap=pipe_gap, delay_step=delay_step)
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.episode = 0
        self.step = 0
        self.env.reset()

    def run(self):
        super(Environment, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()

            state, reward, done, info = self.env.step(action)
            self.step += reward

            if done:
                state = self.reset()
                print(self.episode, self.env_idx, self.step)
                self.step = 0

            self.child_conn.send([state, reward, done, info])

    def reset(self):
        self.steps = 0
        self.episode += 1
        state = self.env.reset()
        return state