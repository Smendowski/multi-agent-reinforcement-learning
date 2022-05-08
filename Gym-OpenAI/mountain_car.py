import pickle
from typing import List, Tuple, Dict

import gym
import numpy as np
import matplotlib.pyplot as plt


class Hyperparameter(dict):
    def __getattr__(self, value):
        return self[value]


def configure_hyperparameters(
    initial_epsilon_value: np.float32, num_of_episodes: int
) -> Hyperparameter:
    actions_mapping = {'MOVE_LEFT': 0, 'NO_MOVEMENT': 1, 'MOVE_RIGHT': 2}

    hyperparameters = Hyperparameter({
        'learning_rate': 0.10,
        'discount_factor': 0.95,
        'epsilon': initial_epsilon_value,
        'num_of_discrete_buckets': 20,
        'num_of_episodes': num_of_episodes,
        'actions': list(actions_mapping.values()),
        'total_rewards': np.zeros(num_of_episodes),
        'start_epsilon_decaying': 1,
        'end_epsilon_decaying': num_of_episodes // 2,
        'epsilon_decay_value':
            initial_epsilon_value / ((num_of_episodes // 2) - 1)
    })

    return hyperparameters


def get_discrete_space(
    lower_bound: np.float32,
    upper_bound: np.float32,
    num_of_discrete_values: int
) -> np.ndarray:
    return np.linspace(
        lower_bound, upper_bound, num_of_discrete_values)


def get_discrete_state(
    observation: List, position_space, velocity_space
) -> Tuple[int, int]:
    position, velocity = observation
    position_bucket = np.digitize(position, position_space)
    velocity_bucket = np.digitize(velocity, velocity_space)

    return position_bucket, velocity_bucket


def create_discrete_states(num_of_buckets) -> List:
    states = []
    for position in range(num_of_buckets + 1):
        for velocity in range(num_of_buckets + 1):
            states.append((position, velocity))

    return states


def initialize_q_table(states: List, actions: List) -> Dict:
    q_table = {}
    for state in states:
        for action in actions:
            q_table[state, action] = 0

    return q_table


def get_best_future_action(q_table: Dict, state, actions: List) -> np.float32:
    values = np.array(
        [q_table[state, action] for action in actions])
    action = np.argmax(values)

    return action


def plot_mean_rewards(
    num_of_games: int, total_rewards: np.ndarray, filename: str
) -> None:
    mean_rewards = np.zeros(num_of_games)
    for episode in range(num_of_games):
        mean_rewards[episode] = np.mean(
            total_rewards[max(0, episode-50):(episode+1)])

    plt.plot(mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Dependency between episodes and scores')
    plt.savefig(filename)


def load_q_learning_table(filename: str) -> Dict:
    return pickle.load(open(filename, 'rb'))


def save_q_learning_table(q_table: Dict, filename: str) -> None:
    if not filename.endswith('pkl'):
        filename = f"{filename.spli('.')[0]}.pkl"

    with open(filename, 'wb') as file:
        pickle.dump(q_table, file)


def main() -> None:
    env = gym.make('MountainCar-v0')
    env.min_speed = -env.max_speed
    env._max_episode_steps = 1000  # Alternatively: 200

    initial_epsilon_value = 1.0  # Alternatively: 0.5
    num_of_episodes = 25_000  # Alternatively: 20_000
    score = 0

    hyperparameter = configure_hyperparameters(
        initial_epsilon_value, num_of_episodes)

    epsilon_bundles = range(
        hyperparameter.start_epsilon_decaying,
        hyperparameter.end_epsilon_decaying
    )

    position_space = get_discrete_space(
        env.min_position, env.max_position,
        hyperparameter.num_of_discrete_buckets)

    velocity_space = get_discrete_space(
        env.min_speed, env.max_speed,
        hyperparameter.num_of_discrete_buckets)

    states = create_discrete_states(
        hyperparameter.num_of_discrete_buckets)

    q_table = initialize_q_table(
        states, hyperparameter.actions)

    for episode in range(hyperparameter.num_of_episodes):
        episode_finished = False
        initial_episode_observation = env.reset()
        initial_episode_state = get_discrete_state(
                initial_episode_observation,
                position_space, velocity_space)

        if episode % 100 == 0 and episode > 0:
            print(f"| Episode: {episode} | Score: {score} |"
                  f"Epsilon: {hyperparameter.epsilon:.3f} |")

        score = 0
        while not episode_finished:
            # env.render()

            # Greedy strategy.
            current_action = np.random.choice(hyperparameter.actions) \
                if np.random.random() < hyperparameter.epsilon \
                else get_best_future_action(
                    q_table, initial_episode_state, hyperparameter.actions)

            current_observation, current_reward, \
                episode_finished, _ = env.step(current_action)

            if current_observation[0] > env.goal_position:
                print(f"Goal reached at {episode} episode")
                # Uncomment to observe when car reaches the destination.
                # env.render()
                # input()
            else:
                current_state = get_discrete_state(
                    current_observation, position_space, velocity_space)

                best_future_action = get_best_future_action(
                    q_table, current_state, hyperparameter.actions)

                # Update Q-Learning table according to the formula.
                old_q_value = q_table[initial_episode_state, current_action]
                future_q_value = q_table[current_state, best_future_action]
                new_q_value = current_reward + \
                    hyperparameter.discount_factor * future_q_value
                tmp_difference = new_q_value - old_q_value

                q_table[initial_episode_state, current_action] = \
                    old_q_value + hyperparameter.learning_rate * tmp_difference

                initial_episode_state = current_state
                score = score + current_reward

        hyperparameter.total_rewards[episode] = score

        # Minimize the greedy selection strategy - update epsilon value.
        if episode in epsilon_bundles:
            if hyperparameter.epsilon > 0.01:
                hyperparameter.epsilon = \
                    hyperparameter.epsilon - hyperparameter.epsilon_decay_value
            else:
                hyperparameter.epsilon = 0.01

    plot_mean_rewards(
        hyperparameter.num_of_episodes,
        hyperparameter.total_rewards,
        'mountain-car.png')

    save_q_learning_table(q_table, 'mountain-car.pkl')


if __name__ == '__main__':
    main()
