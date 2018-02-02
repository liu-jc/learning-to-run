from osim.env import RunEnv
from time import time

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 0)
total_reward = 0.0
start = time()
for i in range(100200):
	each_start = time()
	observation, reward, done, info = env.step(env.action_space.sample())
	total_reward += reward 
	each_end = time()
	print('In {} step,'.format(i) + str(each_end - each_start) + 's,' +' reward is {}'.format(reward))
	if done:
		print(i)
		break
end = time() 

print('total reward {}'.format(total_reward))
print(str(end - start) + 's') 
print(env.action_space)
print(env.observation_space)
