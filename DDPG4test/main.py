from ddpg import *
import tensorflow as tf
import gc
from osim.env import RunEnv
import os
from time import time
gc.enable()

EPISODES = 100000
TEST = 5

def main():
    env = RunEnv(visualize=False)
    agent = DDPG(env)
    latest_ckpt_path = agent.load_model('./saved_model')
    if latest_ckpt_path is None:
        global_step = 0
    else:
        global_step = int(latest_ckpt_path.split('-')[1])
    print('Now, global_step is {}'.format(global_step))
    main_start = time()
    for episode in range(EPISODES):
        current_step = episode + global_step
        start = time()
        state = env.reset()
        #state = env.reset(difficulty = 0)
        #print "episode:",episode
        # Train
        total_reward = 0
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            total_reward += reward
            if done:
                print('After {} episodes, reward is {}'.format(current_step, total_reward))
                break
        end = time()

        if episode % 50 == 0 :
            agent.save_model('./saved_model', current_step)
            main_end = time()
            if not os.path.exists('./logdir'):
                os.mkdir('./logdir')
            with open('./logdir/time.logs','a+') as f:
                f.write('{} episode spent {} seconds.\n'.format(episode, str(end - start)))
                main_hours = (main_end - main_start) / 3600.0
                f.write('Trainging {} episodes spent {} seconds, {} hours.\n'.format(episode + 1, str(main_end - main_start), str(main_hours)))
            total_reward = 0 
            for i in range(TEST):
                state = env.reset()
                for step in range(env.spec.timestep_limit):
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            log_str = 'TESTING: after {} episodes, reward is {}'.format(current_step, total_reward/TEST)
            print(log_str)
            with open('./logdir/rewards.log','a+') as f:
                f.write(log_str + '\n')
   #env.monitor.close()

if __name__ == '__main__':
    main()
