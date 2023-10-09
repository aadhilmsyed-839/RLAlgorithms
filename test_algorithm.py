from config import *

# Define Global Variables
global test_num, results_dict, it, colors, env_name, train_steps, rl_algs

def test_algorithm(alg: Type['abc.ABCMeta'], env_name : str, timesteps : int):

    """
    Description:
        This function takes an SB3 Reinforcement Learning Algorithm as a parameter
        and trains the specified OpenAI Gym environment using this algoirthm. The
        trained model is then executed and a video recording of the agent is saved.

    Parameters:
        - alg (abc.ABCMeta) : Indicates which algorithm to use for training
        - env_name (str)    : Name of the Environment to test in
        - timesteps (int)   : Number of Timesteps for the Learning Stage

    Returns:
        - timesteps (int)      : Number of Timesteps the trained agent performed during execution
        - total_time (float)   : Number of Seconds the Training / Execution Took
        - total_reward (float) : The Total Reward From the Execution Stage of the Trained Agent
    """

    # Start Algorithm Timer
    alg_start = time.time()
    
    # Create environment
    env = None
    try: env = gym.make(env_name, render_mode = "rgb_array")
    except: raise ValueError(f"Unknown Environment: {env_name}")

    # Instantiate the agent
    model = None
    try: model = alg("MlpPolicy", env, verbose = 0)
    except: raise ValueError(f"Unknown Algorithm: {alg}")

    # Set up a new logger
    print("Setting up Logger:")
    log_path = f'./{env_name}/{alg.__name__}/log/'
    formats = ["stdout", "csv", "tensorboard"]
    new_logger = configure(log_path, formats)
    model.set_logger(new_logger)
    
    # Train the agent and display a progress bar
    model.learn(total_timesteps = int(timesteps), progress_bar = True)
    
    # Initialize total reward and timesteps
    total_rew : float = 0
    timesteps : int   = 0
    
    # Create a vectorized environment for recording
    vec_env = VecVideoRecorder(
        venv                 = model.get_env(),
        video_folder         = f'./{env_name}/{alg.__name__}/videos/',
        record_video_trigger = 10_000,
        video_length         = 10_000,
    )
    
    # Execute the Trained Agent
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)
        vec_env.render("rgb_array")
        total_rew += rewards
        timesteps += 1
        if done: break
    
    # Close the vectorized Environment - Stops Recording
    vec_env.close()

    # Stop the Algorithm Timer
    total_time = time.time() - alg_start

    # Return the timesteps, execution time, and reward
    return timesteps, total_time, total_rew[0]

    # Delete Local Variables for Next Iteration
    del model, env, vec_env
