import gym
from IPython import display
from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
from matplotlib import animation





#show_video from https://ymd_h.gitlab.io/ymd_blog/posts/gym_on_google_colab_with_gnwrapper/

def show_video(env, agent = None, dpi = 72, interval = 20):
  d = Display()
  d.start()
  observation = env.reset()

  img = []
  score = 0
  for _ in range(100):
      if agent==None:
        action = env.action_space.sample()
      else:
        observation_tensor = torch.tensor([observation])
        action = agent_trained(observation_tensor)[0][0]
      observation, reward, terminated, truncated , info = env.step(action) # Take action from DNN in actual training.
      display.clear_output(wait=True)
      img.append(env.render('rgb_array'))
      score+=reward
      if terminated or truncated:
          env.reset()
          break

  #dpi = 72
  #interval = 20 # ms

  plt.figure(figsize=(img[0].shape[1]/dpi,img[0].shape[0]/dpi),dpi=dpi)
  patch = plt.imshow(img[0])
  plt.axis=('off')
  animate = lambda i: patch.set_data(img[i])
  ani = animation.FuncAnimation(plt.gcf(),animate,frames=len(img),interval=interval)
  display.display(display.HTML(ani.to_jshtml()))
  return score
  

def query_environment(name):
    env = gym.make(name,new_step_api = True)
    spec = gym.spec(name)
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")

def calc_qvals(rewards,GAMMA):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))