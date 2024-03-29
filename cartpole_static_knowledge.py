import xcs_rc
import gym

agent = xcs_rc.Agent()
agent.reward_map(max=100.0, projected=70.0)
agent.pop.load("cartpole_solved.csv")
env = gym.make('CartPole-v0')
scores = []

for i_episode in range(50):
    state = env.reset()
    for t in range(200):
        env.render()

        input = [state[1], state[2]]
        action = agent.next_action(input, 1)
        state, reward, done, info = env.step(action)

        if done:
            print("Episode #{} finished after {} timesteps".format(i_episode + 1, t + 1))
            scores.append(t + 1)
            if len(scores) > 20:
                scores.pop(0)

            break

print("Average of {} episodes: {}:".format(len(scores), float(sum(scores) / len(scores))))
env.close()
