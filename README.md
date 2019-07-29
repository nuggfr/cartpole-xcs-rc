## OpenAI-Gym CartPole-v0 with XCS-RC

A fully classical Reinforcement Learning solution for CartPole environment from OpenAI-Gym (solved in **38 episodes** so far).

**Links for XCS-RC**
* [PyPI](https://pypi.org/project/xcs-rc/) - `pip install xcs-rc`
* [Github](https://github.com/nuggfr/xcs-rc-python)

---

## Code Description `CartPole-xcs-rc.py`

**Initialization**
```
import xcs_rc
import gym

agent = xcs_rc.Agent(maxpopsize=100, tcomb=10, predtol=20.0, prederrtol=0.0)
agent.explore_it = 0.7
env = gym.make('CartPole-v0')
stop_learning = False
scores = []
```

**Reward Function** (judgement of input-action suitability)
```
def get_reward(input, action):
    reward = 0.0

    if input[1] * (action - 0.5) >= 0.0:
        reward = 0.8
    if abs(input[1]) < 0.05:
        reward += 0.2

    return reward
```

**First State and Action**
```
for i_episode in range(50):
    state = env.reset()
    input = [state[1], state[2]]
    for t in range(200):
        env.render()

        action = agent.next_action(input, 2, 1 - i_episode)
```

**Get `env` Response and Assign `my_reward`**
```
        state, reward, done, info = env.step(action)
        input = [state[1], state[2]]

        if not stop_learning:
            my_reward = agent.maxreward * get_reward(input, action)
            agent.apply_reward(my_reward)
```

**Terminating Current Episode**
```
        if done:
            print("Episode #{} finished after {} timesteps".format(i_episode + 1, t + 1))
            scores.append(t + 1)
```

**Store Knowledge to File, End Current Episode**
```
            save_mode = 'w' if i_episode == 0 else 'a'
            title = "Episode: " + str(i_episode + 1)
            agent.save_popfile("cartpole.csv", title, save_mode)
            break
```
  
**Stop Learning if Knowledge is Solid**
```
    if len(scores) >= 20 and not stop_learning:
        check_stop = scores[-20:]
        if float(sum(check_stop) / 20) >= 195.0:
            print("Learning stopped.")
            stop_learning = True
            sleep(10)  # add little pause here to prepare recording ;)
```

**Terminate if CartPole is Solved**
```
    if len(scores) >= 100:
        check_solved = scores[-100:]
        if float(sum(check_solved) / 100) >= 195.0:
            break
```

**Final Report and Close**
```
agent.combine_pop()
agent.print_pop("\nFinal Population")
agent.save_popfile("cartpole.csv", 'Final', 'a')
print("Average last 100 episodes:", float(sum(scores) / 100))
env.close()
```

---

## Results

Outputs:
* [Console](https://github.com/nuggfr/cartpole-xcs-rc/blob/master/console.txt)
* [Learning results](https://github.com/nuggfr/cartpole-xcs-rc/blob/master/cartpole.csv)
* **Video**

[![CartPole XCS-RC](https://img.youtube.com/vi/mJoavWV80MM/0.jpg)](https://youtu.be/mJoavWV80MM)

Test Code:
* [CartPole.py static knowledge](https://github.com/nuggfr/cartpole-xcs-rc/blob/master/cartpole_static_knowledge.py) (no learning)
