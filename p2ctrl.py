import numpy as np
import random
import serial

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.asarray, zip(*batch))
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

class DDQNN():
    def __init__(self) -> None:
        pass
        
    def build_network(input_dim=4, output_dim=8):
        model = Sequential()
        model.add(Dense(16, input_dim=input_dim, kernel_initializer=he_normal()))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, kernel_initializer=he_normal()))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def act(state, exploration_rate):
        if np.random.rand() < exploration_rate:
            # Exploration: choose a random action
            action = np.random.randint(low=0, high=output_dim)
        else:
            # Exploitation: choose the action with highest Q-value
            q_values = model.predict(np.array([state]))
            action = np.argmax(q_values)
        return action

    def compile_network(model, learning_rate):
        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=opt)

    def evaluate(env, agent, num_episodes=10, max_steps=200):
        """
        Evaluate the agent over multiple episodes on the given environment.

        Args:
            env (gym.Env): The environment to evaluate the agent on.
            agent (DQNAgent): The agent to use for evaluation.
            num_episodes (int): The number of episodes to run evaluation on.
            max_steps (int): The maximum number of steps to take per episode.

        Returns:
            A tuple of mean absolute error, mean squared error, and percentage error.
        """
        mae_list, mse_list, perc_err_list = [], [], []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            step = 0
            mae, mse, perc_err = 0, 0, 0

            while not done and step < max_steps:
                # Select an action
                action = agent.act(state, explore=False)

                # Take a step in the environment
                next_state, reward, done, _ = env.step(action)

                # Update evaluation metrics
                mae += abs(next_state[-1] - env.target_pressure)
                mse += (next_state[-1] - env.target_pressure) ** 2
                perc_err += abs((next_state[-1] - env.target_pressure) / env.target_pressure)

                state = next_state
                step += 1

            mae /= step
            mse /= step
            perc_err /= step

            mae_list.append(mae)
            mse_list.append(mse)
            perc_err_list.append(perc_err)

        mean_mae = sum(mae_list) / len(mae_list)
        mean_mse = sum(mse_list) / len(mse_list)
        mean_perc_err = sum(perc_err_list) / len(perc_err_list)

        return mean_mae, mean_mse, mean_perc_err

    ## UPDATED TRAIN WITH TAKE_ACTION
    def train(env, model, target_model, memory, n_epochs, batch_size, gamma, update_freq, exploration_rate, min_exploration_rate, exploration_decay_rate, rolling_window=10, patience=15):
        best_mae = np.inf
        best_weights = None
        rolling_mae = []
        exploration_step = (exploration_rate - min_exploration_rate) / exploration_decay_rate

        for epoch in range(n_epochs):
            obs = env.reset()
            done = False
            epoch_loss = []

            while not done:
                # Epsilon-greedy exploration strategy
                if np.random.rand() < exploration_rate:
                    action = np.random.randint(env.action_space.n)
                else:
                    action = np.argmax(model.predict(obs[np.newaxis, :]))

                next_obs, reward, done, _ = env.step(action)
                memory.add(obs, action, reward, next_obs, done)
                obs = next_obs

                # Update target network weights
                if memory.counter % update_freq == 0:
                    target_model.set_weights(model.get_weights())

                # Sample from replay memory and train the model
                if memory.counter > batch_size:
                    batch = memory.sample(batch_size)
                    loss = train_step(batch, model, target_model, gamma)
                    epoch_loss.append(loss)

            # Decrease exploration rate
            exploration_rate = max(min_exploration_rate, exploration_rate - exploration_step)

            # Calculate evaluation metrics and print progress
            if epoch % 10 == 0:
                mae, mse, percent_error, ttss = evaluate(model, env)
                rolling_mae.append(mae)

                if len(rolling_mae) > rolling_window:
                    rolling_mae.pop(0)

                current_mae = np.mean(rolling_mae)

                if current_mae < best_mae:
                    best_mae = current_mae
                    best_weights = model.get_weights()
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"Epoch {epoch}/{n_epochs}, Exploration: {exploration_rate:.2f}, MAE: {current_mae:.2f}, MSE: {mse:.2f}, % Error: {percent_error:.2f}, TTSS: {ttss:.2f}, Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping")
                    break

        model.set_weights(best_weights)
        return model


    def receive_data():
        # Initialize serial connection
        ser = serial.Serial('COM3', 115200)
        
        while True:
            # Read a line of data from the serial port
            data = ser.readline().decode().strip()

            # Split the data into individual pressure values
            pressures = data.split(',')
            
            # Convert pressure values from string to integer
            supply_pressure = int(pressures[0])
            chamber_pressure = int(pressures[1])
            atmospheric_pressure = int(pressures[2])
            target_pressure = int(pressures[3])
            
            # TODO: pass the pressure values to the DQNN for action selection and control


    def receiver(port='COM3', baudrate=115200):
        ser = serial.Serial(port=port, baudrate=baudrate)
        while True:
            line = ser.readline().decode('utf-8').rstrip()
            values = list(map(int, line.split(',')))
            values = np.clip(values, 8000, 120000)
            values = (values - 8000) / (120000 - 8000)
            yield values

    def send_control_signal(action):
        with serial.Serial('COM3', 115200, timeout=1) as ser:
            ser.write(bytes([action]))
            ser.write(b'\n')
        return

if __name__ == '__main__':
    pass
