import xcs_rc
import gym
from time import sleep

agent = xcs_rc.Agent(maxpopsize=100, tcomb=20, predtol=20.0, prederrtol=0.0)
agent.reward_map(max=100.0, projected=70.0)
env = gym.make('CartPole-v0')
stop_learning = False
scores = []


def get_reward(input, action):
    reward = 0.0

    if input[1] * (action - 0.5) >= 0.0:
        reward = 0.8
    if abs(input[1]) < 0.05:
        reward += 0.2

    return reward


for i_episode in range(200):
    state = env.reset()
    input = [state[1], state[2]]
    for t in range(200):
        env.render()

        action = agent.next_action(input, 1, 1 - i_episode)

        state, reward, done, info = env.step(action)
        input = [state[1], state[2]]

        if not stop_learning:
            my_reward = agent.maxreward * get_reward(input, action)
            agent.apply_reward(my_reward)

        if done:
            print("Episode #{} finished after {} timesteps".format(i_episode + 1, t + 1))
            scores.append(t + 1)

            save_mode = 'w' if i_episode == 0 else 'a'
            title = "Episode: " + str(i_episode + 1)
            agent.pop.save("cartpole.csv", title, save_mode)
            break

    if len(scores) >= 20 and not stop_learning:
        check_stop = scores[-20:]
        if float(sum(check_stop) / 20) >= 195.0:
            print("Learning stopped.")
            stop_learning = True
            sleep(10)

    if len(scores) >= 100:
        check_solved = scores[-100:]
        if float(sum(check_solved) / 100) >= 195.0:
            break

agent.combine_pop()
agent.print_pop("\nFinal Population")
agent.save_popfile("cartpole.csv", 'Final', 'a')
print("Average last 100 episodes:", float(sum(scores) / 100))
env.close()
