from .base_agent import BaseAgent
from models.ff_model import FFModel
from policies.H_MPC_policy import MPCPolicy
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.utils import *
import math

class HRLAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(HRLAgent, self).__init__()
        envi = env.unwrapped
        self.env = hrlwrapper(envi.unwrapped, agent_params['subpolicies'])
        self.agent_params = agent_params
        self.ensemble_size = self.agent_params['ensemble_size']

        self.dyn_models = []
        for i in range(self.ensemble_size):
            model = FFModel(
                self.agent_params['ac_dim'],
                self.agent_params['ob_dim'],
                self.agent_params['n_layers'],
                self.agent_params['size'],
                self.agent_params['learning_rate'],
            )
            self.dyn_models.append(model)

        self.actor =MPCPolicy(
            self.env,
            ac_dim=self.agent_params['ac_dim'],
            dyn_models=self.dyn_models,
            horizon=self.agent_params['mpc_horizon'],
            N=self.agent_params['mpc_num_action_sequences'],
            sample_strategy=self.agent_params['mpc_action_sampling_strategy'],
            cem_iterations=self.agent_params['cem_iterations'],
            cem_num_elites=self.agent_params['cem_num_elites'],
            cem_alpha=self.agent_params['cem_alpha'],
        )

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        for i in range(self.ensemble_size):

            # select which datapoints to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful
            ens_top_idx = (i+1) * num_data_per_ens
            ens_bottom_idx = i * num_data_per_ens
            observations = ob_no[ens_bottom_idx: ens_top_idx]
            actions = ac_na[ens_bottom_idx: ens_top_idx]
            next_observations = next_ob_no[ens_bottom_idx: ens_top_idx]

            # use datapoints to update one of the dyn_models
            model = self.dyn_models[i]
            log = model.update(observations, actions, next_observations,
                                self.data_statistics)
            loss = log['Training Loss']
            losses.append(loss)

        avg_loss = np.mean(losses)
        return {
            'Training Loss': avg_loss,
        }

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            'obs_mean': np.mean(self.replay_buffer.obs, axis=0),
            'obs_std': np.std(self.replay_buffer.obs, axis=0),
            'acs_mean': np.mean(self.replay_buffer.acs, axis=0),
            'acs_std': np.std(self.replay_buffer.acs, axis=0),
            'delta_mean': np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
                                'delta_std': np.std(
                                    self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0),
                                }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = self.data_statistics

    def sample(self, batch_size):
        # NOTE: sampling batch_size * ensemble_size,
        # so each model in our ensemble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(
            batch_size * self.ensemble_size)


import gym 
from gym import spaces
import torch
class hrlwrapper(gym.Wrapper): 
    def __init__(self, env, subpolicies):
        super().__init__(env)
        # loadmodel = 'data/mb_cem_i4_e5_s250_l2_spacecraft-docking-continuous-v0_08-10-2022_16-23-08/mlp_model_itr_24'
        # loadmodel2 = 'data/mb_rand_10_1000_spacecraft-docking-continuous-v0_08-10-2022_03-23-23/mlp_model_itr_24'
        self.phase2 = torch.load(subpolicies[0])
        self.phase3 = torch.load(subpolicies[1])
        self.subpolicies = [self.phase2, self.phase3]
        self.env = env
        self.action_space = spaces.MultiDiscrete([2, 5]) # 2 options, max 5 seconds between decisions
        self.max_path_length = 4000

    def step(self, action, obs):
        # action will be of the form [option, tau]
        option, tau = action
        tau=tau+1 # tau cannot be 0
        subpolicy = self.subpolicies[option]
        steps = 0 
        reward = 0
        self.changephase(int(option+2))
        for t in range(tau):
            t=t+5
            act = subpolicy.get_action(obs)
            obs_old = obs.copy()
            obs, rew, done, _= self.env.step(act)
            reward += rew['rew']
            steps += 1
            rollout_done = done or (steps >= self.max_path_length)
            # terminals.append(rollout_done)
            if rollout_done:
                break
            
        hrew, hdone, hinfo = self.get_Hreward(obs, obs_old, steps, reward)
        info = {'option': option, 'tau': tau, 'steps':steps}
        hinfo['rew'] = reward+hrew

        return obs, hinfo, hdone, info

    def get_Hreward(self, obs, obs_old, hstep, reward, actions=None):
        '''calculates reward in mpc function 
        Args:
            obs (nparray): array of obs
            actions (nparray): array of actions
            hstep (int): mpc horizon step
        Returns:
            rewards this step and done conditions: rew and done
        '''

        if (len(obs.shape) == 1):
            obs = np.expand_dims(obs, axis=0)
            # actions = np.expand_dims(actions, axis=0)
            batch_mode = False
        else: 
            batch_mode = True
        if (len(obs_old.shape) == 1):
            obs_old = np.expand_dims(obs_old, axis=0)

        if obs_old[0][0] == None: 
            obs_old = self.state

        if (len(obs_old.shape) == 1):
            obs_old = np.expand_dims(obs_old, axis=0)
        
        xpos      = obs[:, 0]
        ypos      = obs[:, 1]  # obs
        x_dot_obs = obs[:, 2]
        y_dot_obs = obs[:, 3]

        # if hstep == 1 or hstep == 0: 
            # self.hcinput = self.env.control_input

        rho = np.linalg.norm([xpos,ypos], axis=0)
        # x_force = actions[:, 0]
        # y_force = actions[:, 1]
        # self.hcinput += (abs(x_force) + abs(y_force)) * self.TAU

        vH = np.linalg.norm([x_dot_obs, y_dot_obs], axis=0)  # Velocity Magnitude

        # check dones conditions
        dones = np.zeros((obs.shape[0],))
        dn1 = np.zeros((obs.shape[0],))
        dn2 = np.zeros((obs.shape[0],))
        dn1[abs(ypos) <= self.pos_threshold] = 1
        dn2[abs(xpos) <= self.pos_threshold] = 1
        dones[dn1+dn2>1] = 1 # if s/c is within docking bounds

        # dones[abs(xpos) > self.x_threshold] = 1
        # dones[abs(ypos) > self.y_threshold] = 1
        # dones[self.hcinput > np.ones((obs.shape[0],))*self.max_control] = 1  # upper bound on fuel exceeded
        dones[(self.steps+hstep) * self.TAU >= self.max_time-1] = 1

        #calc rewards
        # reward = np.zeros((obs.shape[0],))
        failure  = np.zeros((obs.shape[0],))    
        success  = np.zeros((obs.shape[0],))    
        crash    = np.zeros((obs.shape[0],))    
        overtime = np.zeros((obs.shape[0],))    
    
        # if not dones: 
        c = np.zeros([obs.shape[0], 3])
        c[:, 1] = 100
        position = np.array([xpos, ypos, np.zeros(obs.shape[0])]).transpose()
        val = position[:, 1] * 100 / (100 * rho)  # dot(position, c) / mag(position)*mag(c)

        if ~all(dones):    
            # # get within LoS sooner during phase 2
            reward[((dones==0) & (rho < 1000) & (val <= math.cos(self.env.theta_los)) & (ypos > obs_old[:, 1]))] += 1.5
            # be contious of fuel consumption
            # reward[((dones==0) & (abs(self.hcinput)>0))] += -self.hcinput/1000
            # reward[((dones==0) & (rho <= 100) & (val <= math.cos(self.theta_los)))] += 1 #! from phase 2 rew
            # reward[((dones==0) & (rho <= 200) & (val <= math.cos(self.theta_los)) & (ypos > obs_old[:, 1]) ) ] += .5 #! from phase 2 rew

        if all(dones) != False: 
            reward[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH > self.VEL_THRESH))] += -0.001
            reward[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH <= self.VEL_THRESH))] += 1
            reward[((dones==1) & ((self.steps+hstep) * self.TAU > self.max_time))] += -1
            reward[((dones==1) & (abs(xpos) > self.x_threshold) & (abs(ypos) > self.y_threshold))] += -1

            success[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH <= self.VEL_THRESH))] = 1
            crash[((dones==1) & (abs(xpos) <= self.pos_threshold) & (abs(ypos) <= self.pos_threshold) & (vH > self.VEL_THRESH))] = 1
            failure[((dones==1) & (abs(xpos) > self.pos_threshold) & (abs(ypos) > self.pos_threshold))] = 1
            overtime[((dones==1) & ((self.steps+hstep) * self.TAU >= self.max_time-1))] = 1

        info = {}
        info['success']  = sum(success)
        info['crash']    = sum(crash)
        info['failure']  = sum(failure)
        info['overtime'] = sum(overtime)

        return reward, dones, info

def H_Path(obs, acs, rewards, next_obs, terminals, option, tau, crashes, successes, failures, overtimes):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation": np.array(obs, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "options": np.array(option, dtype=int),
        "tau": np.array(tau, dtype=int),
        "crashes": np.array(crashes, dtype=np.float32),
        "successes": np.array(successes, dtype=np.float32),
        "failures": np.array(failures, dtype=np.float32),
        "overtimes": np.array(overtimes, dtype=np.float32)
        }