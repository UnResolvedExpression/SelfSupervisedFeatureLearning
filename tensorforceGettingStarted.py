import numpy as np
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

class CustomEnvironment(Environment):

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='float', shape=(8,))

    def actions(self):
        return dict(type='int', num_values=4)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(8,))
        return state

    def execute(self, actions):
        next_state = np.random.random(size=(8,))
        terminal = False  # Always False if no "natural" terminal state
        reward = np.random.random()
        return next_state, terminal, reward

def main():
    # environment = Environment.create(
    #     environment='gym', level='CartPole', max_episode_timesteps=500)
    environment = CustomEnvironment()
    agent = Agent.create(
        agent='tensorforce', environment=environment, update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(horizon=20)
    )
    # Train for 100 episodes
    for _ in range(5):
        print(" train " + str(_))

        states = environment.reset()
        terminal = False
        counter=0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            counter+=1
            if counter % 100 == 0:
                print('counter ' + str(counter))
                print(actions)
                print(states)
                print(reward)
    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(5):
        print(_)
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        counter=0
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals,
                independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
            counter+=1
            if counter%100==0:
                print(actions)
                print(states)
                print(reward)

    print('Mean episode reward:', sum_rewards / 100)

    # Close agent and environment
    agent.close()
    environment.close()




if __name__ == '__main__':
    print("hi")
    main()


