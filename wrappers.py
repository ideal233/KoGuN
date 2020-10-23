import gym
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
import numpy as np
FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400
class LunarLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=3000):
        super(LunarLanderWrapper, self).__init__(env)
        if max_episode_steps is None:
            max_episode_steps = env.spec.max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.state_his = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        assert self.state_his is not None, "Cannot call env.step() before calling reset()"
        state_his = self.state_his
        state, reward, done, info = self.env.step(action)
        self.state_his = state
        self._elapsed_steps += 1

        import math
        reward1 = (math.fabs(state_his[0]) - math.fabs(state[0]))
        reward1 += (math.fabs(state_his[1]) - math.fabs(state[1]))
        reward1 += (state_his[2] ** 2 + state_his[3] ** 2) - (state[2] ** 2 + state[3] ** 2)
        reward1 += (math.fabs(state_his[4]) - math.fabs(state[4]))

        info['true_r'] = reward
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
            # if self.env.lander.awake:
            #     reward = -100
        return state, reward1, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.state_his = self.env.reset(**kwargs)
        return self.state_his

class AscentYLunarLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=3000):
        super(AscentYLunarLanderWrapper, self).__init__(env)
        if max_episode_steps is None:
            max_episode_steps = env.spec.max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.update_count = 0
        self.step_count = 0


    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"


        state, reward, done, info = self.env.step(action)
        self.step_count += 1
        if self.step_count % 128 == 0:
            self.update_count += 1
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
            # if self.env.lander.awake:
            #     reward = -100
        return state, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        obs = self.env.reset(**kwargs)
        import random
        start_y = 4.5
        end_y = VIEWPORT_H/SCALE
        current_y = start_y + self.update_count/10000.0 * (end_y - start_y)
        initial_y = random.uniform(start_y, current_y)
        self.env.lander = self.env.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.env.lander.color1 = (0.5, 0.4, 0.9)
        self.env.lander.color2 = (0.3, 0.3, 0.5)
        self.env.lander.ApplyForceToCenter((
            self.env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.env.legs = []
        for i in [-1, +1]:
            leg = self.env.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.env.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.env.world.CreateJoint(rjd)
            self.env.legs.append(leg)

        self.env.drawlist = [self.env.lander] + self.env.legs

        return self.env.step(np.array([0, 0]) if self.env.continuous else 0)[0]

class MaxMLunarLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_m = 150, max_episode_steps=1000):
        super(MaxMLunarLanderWrapper, self).__init__(env)
        if max_episode_steps is None:
            max_episode_steps = env.spec.max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.max_m = max_m

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        if action == 2:
            self.current_m += 1
            if self.current_m > self.max_m:
                action = 0
        state, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return state, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        self.current_m = 0
        return self.env.reset(**kwargs)