import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import utils.ppo_core as core
from utils.utils import EpochLogger,setup_logger_kwargs, setup_pytorch_for_mpi, sync_params
from utils.utils import mpi_avg_grads, mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from env.quadsim_env import QuadSimEnv
import argparse
# import wandb
import matplotlib.pyplot as plt
import random
import seaborn as sns

sns.set_style('darkgrid')
# export PATH="/home/rich/miniconda3/bin:$PATH"s

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        # print("PTR: {}, Max_size: {}".format(self.ptr, self.max_size))
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-3)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

def learning_curve_display(epoch, last_show_num, logger, eval_rew_list):
    mean_reward = np.mean(logger.epoch_dict['EpRet'])
    eval_rew_list.append(mean_reward)
    if epoch / last_show_num > 1.05:
        plt.cla()
        plt.plot(eval_rew_list, label="Rewards")
        plt.legend()
        plt.pause(0.01)
        last_show_num = epoch
    
    return eval_rew_list, last_show_num

def ppo(env_fn, 
        actor_critic=core.MLPActorCritic, 
        ac_kwargs=dict(), 
        seed=0, 
        steps_per_epoch=50000,
        epochs=5000, 
        gamma=0.99, 
        clip_ratio=0.2, 
        pi_lr=3e-4,
        vf_lr=3e-4, 
        train_pi_iters=5, 
        train_v_iters=5, 
        lam=0.97,
        update_per_epoch = 3,
        max_ep_len=10000,
        target_kl=0.01,
        logger_kwargs=dict(),
        external_distrub = False,
        ):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    
    ctrl_freq = env_fn.ctrl_freq
    sim_freq = env_fn.sim_freq
    ctrl_every_sim_step = int(sim_freq / ctrl_freq)

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed*1)
    np.random.seed(seed*2)
    random.seed(seed*3)

    # Instantiate environment
    obs_dim = env_fn.observation_space.shape
    act_dim = env_fn.action_space.shape

    # Create actor-critic module
    actor_critic = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    logger.setup_pytorch_saver(actor_critic)

    # Sync params across processes
    sync_params(actor_critic)

    # Count variables
    var_counts_1 = tuple(core.count_vars(module) for module in [actor_critic.pi, actor_critic.v])
    logger.log('\nNumber of parameters of Main Quad: \t pi: %d, \t v: %d\n'%var_counts_1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Start train model on: {}'.format(device))

    # Set up experience buffer
    buffer_size = int(steps_per_epoch / ctrl_every_sim_step)
    buffer = PPOBuffer(obs_dim, act_dim, buffer_size, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, actor_critic, entropy_coef=0.01):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Entropy regularization
        entropy = pi.entropy().mean()
        loss_pi -= entropy_coef * entropy

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = entropy.item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data, actor_critic):
        obs, ret = data['obs'], data['ret']
        return ((actor_critic.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(actor_critic.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(actor_critic.v.parameters(), lr=vf_lr)
    
    # Set up model saving
    logger.setup_pytorch_saver(actor_critic)

    def update(buffer, actor_critic, pi_optimizer, vf_optimizer):
        data = buffer.get()
        pi_l_old, pi_info_old = compute_loss_pi(data, actor_critic)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data, actor_critic).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data, actor_critic)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(actor_critic.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data, actor_critic)
            loss_v.backward()
            mpi_avg_grads(actor_critic.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(
                    LossPi=pi_l_old, 
                    LossV=v_l_old,
                    KL=kl, 
                    Entropy=ent, 
                    ClipFrac=cf,
                    DeltaLossPi=(loss_pi.item() - pi_l_old),
                    DeltaLossV=(loss_v.item() - v_l_old))
            
    last_show_num = 1
    eval_rew_list = []

    # Prepare for interaction with environment
    start_time = time.time()
    obs = env_fn.reset()
    ep_ret, ep_len = 0, 0

    '''
    posHistory       = np.zeros([max_ep_len, 3])
    velHistory       = np.zeros([max_ep_len, 3])
    angVelHistory    = np.zeros([max_ep_len, 3])
    attHistory       = np.zeros([max_ep_len, 3])
    motForcesHistory = np.zeros([max_ep_len, env_fn.quadcopter.get_num_motors()])
    times            = np.zeros([max_ep_len, 1])
    '''

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        
        hl_reward = 0
        external_force = np.array([0.0, 0.0, 0.0])
        external_torque = np.array([0.0, 0.0, 0.0])
        # Randomize the start time for x, y, z direction forces
        start_times = {
            'x': random.randint(2500, 3500),
            'y': random.randint(2500, 3500),
            'z': random.randint(2500, 3500),
            'torque': random.randint(2500, 3500),
        }
        period = {
            'x': np.random.uniform(2, 8),
            'y': np.random.uniform(2, 8),
            'z': np.random.uniform(2, 8),
        }
        amplitude = {
            'x': np.random.uniform(0, 2),
            'y': np.random.uniform(0, 2),
            'z': np.random.uniform(0, 6),
        }

        external_end_t = (env_fn.hover_time + env_fn.takeoff_time) * env_fn.sim_freq

        for t in range(steps_per_epoch):
            
            if ep_len % ctrl_every_sim_step == 0:
                action, value, logp = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))
                fix_obs = obs
            
            # ---------------------------------- Define external forces -------------------------------------
            current_time = ep_len / sim_freq  # Calculate current real time in seconds
            if external_distrub:
                external_x, external_y, external_z = 0, 0, 0
                if ep_len > start_times['x'] and ep_len < external_end_t:
                    noise_x = np.random.normal(0, 0.2)
                    external_x = np.clip((amplitude['x'] * np.sin(2 * np.pi * current_time / period['x']) + noise_x), -1, 1)
                
                if ep_len > start_times['y'] and ep_len < external_end_t:
                    noise_y = np.random.normal(0, 0.2)
                    external_y = np.clip((amplitude['y'] * np.sin(2 * np.pi * current_time / period['y']) + noise_y), -1, 1)
                
                if ep_len > start_times['z'] and ep_len < external_end_t:
                    noise_z = np.random.normal(0, 0.2)
                    external_z = np.clip((amplitude['z'] * np.sin(2 * np.pi * current_time / period['z']) + noise_z), -4, 4)

                external_force = np.array([external_x, external_y, external_z])

                if ep_len > start_times['torque'] and ep_len < external_end_t and ep_len % ctrl_every_sim_step == 0:
                    external_torque += np.random.normal([0.0, 0.0, 0.0], [0.02, 0.02, 0.02])
                    external_torque = np.clip(external_torque, [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1])
                else:
                    external_torque = np.array([0.0, 0.0, 0.0])
            # ---------------------------------- Define external forces -------------------------------------
            '''
            times[ep_len] = t
            posHistory[ep_len,:]       = env_fn.quadcopter._pos.to_list()
            velHistory[ep_len,:]       = env_fn.quadcopter._vel.to_list()
            attHistory[ep_len,:]       = env_fn.quadcopter._att.to_euler_YPR()
            angVelHistory[ep_len,:]    = env_fn.quadcopter._omega.to_list()
            motForcesHistory[ep_len,:] = env_fn.quadcopter.get_motor_forces()
            '''

            next_obs, reward, done = env_fn.step(action, ep_len)
            hl_reward += reward / ctrl_every_sim_step
            
            ep_ret += reward / ctrl_every_sim_step
            ep_len += 1

            # save and log
            if ep_len % ctrl_every_sim_step == int(ctrl_every_sim_step-1):
                buffer.store(fix_obs, action, hl_reward, value, logp)
                logger.store(VVals=value)
                hl_reward = 0
            
            # Update obs (critical!)
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = (t == steps_per_epoch - 1)

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, value, _ = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value = 0
                buffer.finish_path(value)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs = env_fn.reset()
                ep_ret, ep_len = 0, 0
                # last_action = np.array([0, 0, 0, 0])

        # Save the best model
        if epoch > 0:
            if np.mean(logger.epoch_dict['EpRet']) > max(eval_rew_list):
                print('Find the Best Performance Model !!!')
                logger.save_state({'env': env}, str_info='best')
                dummy_input = torch.zeros(obs_dim)
                torch.onnx.export(actor_critic.pi.mu_net, dummy_input, "./data/model/{}.onnx".format(args.exp_name+str(args.seed)), verbose=True, input_names=['input'], output_names=['output'])
            logger.save_state({'env': env})
            print('The latest model is saved.')
        
        # Perform PPO update!
        for param in actor_critic.parameters():
            torch.nn.utils.clip_grad_norm_(param, max_norm=0.5)
            
        for update_i in range(update_per_epoch):
            update(buffer, actor_critic, pi_optimizer, vf_optimizer)
        eval_rew_list, last_show_num = learning_curve_display(epoch, last_show_num, logger, eval_rew_list)

        
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        
        logger.dump_tabular()

        '''
        fig, ax = plt.subplots(5,1, sharex=True)

        ax[0].plot(times, posHistory[:,0], label='x')
        ax[0].plot(times, posHistory[:,1], label='y')
        ax[0].plot(times, posHistory[:,2], label='z')
        ax[1].plot(times, velHistory)
        ax[2].plot(times, attHistory[:,0]*180/np.pi, label='Y')
        ax[2].plot(times, attHistory[:,1]*180/np.pi, label='P')
        ax[2].plot(times, attHistory[:,2]*180/np.pi, label='R')
        ax[3].plot(times, angVelHistory[:,0], label='p')
        ax[3].plot(times, angVelHistory[:,1], label='q')
        ax[3].plot(times, angVelHistory[:,2], label='r')
        ax[4].plot(times, motForcesHistory,':')

        ax[-1].set_xlabel('Time [s]')

        ax[0].set_ylabel('Pos')
        ax[1].set_ylabel('Vel')
        ax[2].set_ylabel('Att [deg]')
        ax[3].set_ylabel('AngVel (in B)')
        ax[4].set_ylabel('MotForces')

        ax[0].legend()
        ax[2].legend()
        ax[3].legend()

        plt.show()
        '''

if __name__ == '__main__':
    
    env = QuadSimEnv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=env)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--episode_len', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='proxfly')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    # wandb.init(project="real_world_learning", name=f"ppo-residual-ll", entity="hiperlab")
    
    ppo(env_fn=args.env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hidden_dim]*args.layers), 
        gamma=args.gamma, 
        seed=args.seed, 
        steps_per_epoch=args.steps, 
        epochs=args.epochs,
        max_ep_len=args.episode_len,
        logger_kwargs=logger_kwargs)