from config import *

# Define Global Variables
global test_num, results_dict, it, colors, env_name, train_steps, rl_algs

def generate_results():

    """
    Description:
        This function takes a list of algorithms that have been trained and tested and compares them by printing out the results
        and by plotting them on a scatter plot.

    Parameters:
        None

    Returns:
        - plt (pyplot) : A plot comparing the performance of all the RL algorithms
    """
    
    for alg in rl_algs:
        
        # Print the Result of Each Function
        print(f"{alg.__name__}:\n{results_dict[alg.__name__]}\n")

        # Read the Progress CSV file for each algorithm
        filename = f'./{env_name}/{alg.__name__}/log/progress.csv'
        df = pd.read_csv(filename)

        # Extract the Necessary Columns
        x = df['time/total_timesteps']
        y = df['rollout/ep_rew_mean']

        # Add the Data to the Plot
        plt.plot(x, y, label = alg.__name__, color = colors[it], linestyle='-', marker='o')
        it = ((it + 1) % len(colors))

    # Add Labels and Title to the Plot
    plt.xlabel('Total Timesteps')
    plt.ylabel('Total Mean Reward')
    plt.title('Algorithm Performance for Lunar Lander Environment')
