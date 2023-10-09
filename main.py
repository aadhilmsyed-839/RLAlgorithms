from config import *
from test_algorithm import *
from results import *

# Start Test Counter & Start the Program Timer
prog_start = time.time()

# Test each RL algorithm with the defined parameters
for alg in rl_algs:

    # Program Status Output
    print("=============================================")
    print(f"Starting Test #{test_num}")
    print(f"   Environment: {env_name}")
    print(f"   Algorithm:   {alg.__name__}")
    print(f"   Timesteps:   {int(train_steps)}")
    print("=============================================")
    
    # Test the Algorithm with the defined parameters
    total_timesteps, total_time, total_reward = test_algorithm(
        alg       = alg,
        env_name  = env_name,
        timesteps = train_steps
    )

    # Add the Results to the Results Dictionary
    results_dict[alg.__name__] = (f"--------------------\n"
            f"The {alg.__name__} algorithm achieved a total reward of {total_reward:.2f} in {total_timesteps} timesteps.\n"
            f"The {alg.__name__} algorithm took {total_time:.2f} s ({(total_time/60):.2f} min) to execute.\n")

    # Print the Results
    print(results_dict[alg.__name__])

    # Increment the Test Number to start new iteration
    test_num += 1

# Stop the Program Timer
tot_time = time.time() - prog_start

# Print the Results Dictionary HEADER
print("\n=============================================")
print("                Final Results                ")
print("=============================================\n")

# Get the Generated Results & Export the Plot as an Image
plot = generate_results()
plot.savefig('images/rl_algs.png', dpi=300, bbox_inches='tight')

# Print the Termination Message
print("\n=============================================")
print(f"This program took {(tot_time/60):.2f} mins to execute.\n")
print("Terminating Program...")
print("=============================================\n")
