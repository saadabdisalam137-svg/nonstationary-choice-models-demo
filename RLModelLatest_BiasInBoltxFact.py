import os
import pandas as pd
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def query0():
    return get_choice("Select data type:", {"1": "real", "2": "simulated"})

def query1(data_type):
    return get_choice("Select protocol:", {"1": "rev", "2": "var"})

def query_states():
    return [s.strip() for s in input("Enter states (comma-separated): ").strip().split(",") if s.strip()]

def query_actions():
    return [a.strip() for a in input("Enter actions (comma-separated): ").strip().split(",") if a.strip()]

def query_reward_probs(states, actions):
    reward_probs, correct = {}, {}
    print("\nEnter reward probabilities (0–1) for each state-action pair:")
    for state in states:
        for action in actions:
            while True:
                try:
                    val = float(input(f"P(reward | state={state}, action={action}): ").strip())
                    if 0 <= val <= 1:
                        reward_probs[(state, action)] = val
                        if val > 0.5:
                            correct[(state, action)] = 1
                        break
                    else:
                        print("Please enter a value between 0 and 1.")
                except ValueError:
                    print("Invalid input. Enter a numeric value.")
    return reward_probs, correct

def query2(data_type, protocol):
    return get_choice("Select mouse group:" if data_type == "real" else "Select simulation model type:",
                      {"1": "16p" if data_type == "real" else "constant", "2": "WT" if data_type == "real" else "dynamic"})

def get_choice(prompt, options):
    print(prompt)
    for key, value in options.items():
        print(f"{key}: {value}")
    while True:
        choice = input("Enter your choice: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid choice. Please try again.")
# ----------------------------------------
# FILE MAPPINGS FOR REAL DATA
# ----------------------------------------

links_16p_rev = [
    'mouse_data_6149_16p11.2_rev_prob.csv',
    'mouse_data_6212_16p11.2_rev_prob.csv',
    'mouse_data_6213_16p11.2_rev_prob.csv',
    'mouse_data_6244_16p11.2_rev_prob.csv',
    'mouse_data_6344_16p11.2_rev_prob.csv'
]

links_WT_rev = [
    'mouse_data_6150_WT_rev_prob.csv',
    'mouse_data_6210_WT_rev_prob.csv',
    'mouse_data_6211_WT_rev_prob.csv',
    'mouse_data_6240_WT_rev_prob.csv',
    'mouse_data_6242_WT_rev_prob.csv'
]

links_WT_var = [
    'mouse_data_6571_WT_var_prob.csv',
    'mouse_data_6558_WT_var_prob.csv',
    'mouse_data_6560_WT_var_prob.csv',
    'mouse_data_6564_WT_var_prob.csv',
    'mouse_data_6565_WT_var_prob.csv',
    'mouse_data_6566_WT_var_prob.csv',
    'mouse_data_6723_WT_var_prob.csv'
]

links_16p_var = [
    'mouse_data_6557_16p11.2_var_prob.csv',
    'mouse_data_6561_16p11.2_var_prob.csv',
    'mouse_data_6562_16p11.2_var_prob.csv',
    'mouse_data_6563_16p11.2_var_prob.csv',
    'mouse_data_6569_16p11.2_var_prob.csv',
    'mouse_data_6722_16p11.2_var_prob.csv',
    'mouse_data_6725_16p11.2_var_prob.csv',
    'mouse_data_6730_16p11.2_var_prob.csv',
    'mouse_data_6732_16p11.2_var_prob.csv',
    'mouse_data_6735_16p11.2_var_prob.csv'
]

# ----------------------------------------
# MOUSE DATA LOADER CLASS
# ----------------------------------------


class MouseDataLoader:
    def __init__(self, data_type, protocol, group_or_model, base_path="/Volumes/SacadsProjects/ProjectAutism/ExperimentalData"):
        
        self.data_type = data_type
        self.protocol = protocol
        self.group_or_model = group_or_model
        self.base_path = base_path
        self.links = self.get_file_links()
        self.mouse_num = int(input(f"Select from {0} to {len(self.links)-1} from the available files\n"))
        if not self.links:
            raise ValueError("No data files found for the selected configuration.")
        self._df_cache = None
    def get_file_links(self):
        if self.data_type == "real":
            if self.protocol == "rev":
                return links_16p_rev if self.group_or_model == "16p" else links_WT_rev
            elif self.protocol == "var":
                return links_16p_var if self.group_or_model == "16p" else links_WT_var
        else:
            directory = os.path.join(self.base_path, self.data_type, self.protocol, self.group_or_model)
            return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")] if os.path.isdir(directory) else []

    def load_data(self):
        if not (0 <= self.mouse_num < len(self.links)):
            raise IndexError("Index out of range.")
        file_path = os.path.join(self.base_path, self.links[self.mouse_num]) if self.data_type == "real" else self.links[self.mouse_num]
        print(f"✅ Loading file: {file_path}")
        if self._df_cache is None:
            self._df_cache = pd.read_csv(file_path)
        return self._df_cache

    def get_learning_phases(self):
        df = self.load_data()
        countdown = df['countdown'].tolist()

        # Reversal trials (do NOT include 0)
        self.reversal_trials = [i+1 for i, c in enumerate(countdown) if c == 0 and i != 0]
        final_trial = len(countdown) - 1

        # Build Phases using reversal trials
        Phases = []
        prev, final_phase_expertness = 0,0
        for rev in self.reversal_trials:
            Phases.append([prev, rev - 1])
            prev = rev
        if prev < final_trial:
            Phases.append([prev, final_trial])  # Final phase goes to the end
            final_phase_expertness = 1 # reversal is not in last window of the last phase
        return Phases, final_phase_expertness, self.reversal_trials
    



# ----------------------------------------
# SYNTHETIC DATA GENERATOR (REVERSAL MODEL)
# ----------------------------------------

from collections import defaultdict
import numpy as np
import random

class SyntheticDATA_Generator_constantModel_rev:
    """
    Simulator aligned with your estimator:
      - Parameters: a (alpha), b (beta), biases b_L, b_R, b_N
      - Actions: supports ['L','R'] or ['L','R','N'] (order respected)
      - Policy: softmax( b * Q(a,s) + bias_a )
      - Q-update: Q[a,s] <- (1-a)*Q[a,s] + a*reward
      - Reversal: trigger when last 20 trials have >=19 correct; then after
                  'reversal_countdown' trials, flip reward probs (N fixed to 0)
      - Tracks: states, actions, rewards, q_evolution, total NLL, reversal_trials
    """
    def __init__(self, a, b, b_L, b_R, b_N,
                 trials,
                 past_states, past_actions, past_rewards,
                 state_space, action_space,
                 reward_probs, correct_pairs,
                 reversal_countdown=250, perf_window=20, perf_threshold=19):
        # model params
        self.a = float(a)
        self.b = float(b)
        self.b_L = float(b_L)
        self.b_R = float(b_R)
        self.b_N = float(b_N)

        # simulation config
        self.trials = int(trials)
        self.state_space  = list(state_space)
        self.action_space = list(action_space)  # e.g., ['L','R'] or ['L','R','N']

        # environment mapping
        self.reward_probs  = dict(reward_probs)   # {(state, action) -> p}
        self.correct_pairs = dict(correct_pairs)  # {(state, action) -> 1/0}

        # optional history prefix
        self.past_states  = list(past_states or [])
        self.past_actions = list(past_actions or [])
        self.past_rewards = list(past_rewards or [])

        # tracking
        self.state_counts = {s: 0 for s in self.state_space}
        self.action_counts = {a: 0 for a in self.action_space}
        self.state_action_counts = {(s, a): 0 for s in self.state_space for a in self.action_space}
        self.rewards = defaultdict(list)
        self.q_evolution = []
        self.nll = 0.0
        self.last_qs = None
        self.reversal_trials = []

        # reversal logic
        self.reversal_countdown_init = int(reversal_countdown)
        self.perf_window = int(perf_window)
        self.perf_threshold = int(perf_threshold)

        # bias map by action label
        self._bias_map = {
            'L': self.b_L,
            'R': self.b_R,
            'N': self.b_N
        }

    # --- helpers ---
    def _initialize_q_table(self, previous_q=None):
        if previous_q is None:
            return {(a, s): 0.0 for a in self.action_space for s in self.state_space}
        # ensure full coverage
        q = {(a, s): 0.0 for a in self.action_space for s in self.state_space}
        q.update({k: float(v) for k, v in (previous_q or {}).items() if k in q})
        return q

    def _reward_gen(self, action, state):
        prob = float(self.reward_probs.get((state, action), 0.0))
        r = 1 if random.random() < prob else 0
        self.rewards[(action, state)].append(r)
        return r

    def _softmax_policy(self, q_table, state):
        """
        Compute softmax over actions present in self.action_space
        using utilities u(a) = b*Q[a,s] + bias[a].
        """
        eps = 1e-12
        utils = []
        for a in self.action_space:
            bias = float(self._bias_map.get(a.upper(), 0.0))
            u = self.b * float(q_table[(a, state)]) + bias
            utils.append(u)
        u = np.array(utils, dtype=float)
        u = u - np.max(u)  # stability
        e = np.exp(u)
        p = e / (np.sum(e) + eps)
        return p  # aligned to self.action_space order

    def _reverse_rewards(self):
        """
        Flip reward probabilities for all (state, action) except:
          - Keep 'N' (or 'n') at 0
        Also update correct_pairs accordingly.
        """
        new_reward_probs = {}
        new_correct = {}
        for (s, a), old_prob in self.reward_probs.items():
            if a.upper() == 'N':
                new_reward_probs[(s, a)] = 0.0
                new_correct[(s, a)] = 0
            else:
                new_p = 1.0 - float(old_prob)
                new_reward_probs[(s, a)] = new_p
                new_correct[(s, a)] = 1 if new_p > 0.5 else 0
        self.reward_probs = new_reward_probs
        self.correct_pairs = new_correct
        print("\n🔄 Rule reversal applied!")

    # --- main simulate ---
    def generate(self):
        total_trials = len(self.past_states) + self.trials

        # assemble state/action/reward trajectories
        self.states  = self.past_states + [random.choice(self.state_space) for _ in range(self.trials)]
        self.actions = self.past_actions + [None] * self.trials
        self.Rewards = self.past_rewards + [None] * self.trials

        q_table = self._initialize_q_table(previous_q=None)
        last_k_correct = []
        reversal_triggered = False
        countdown = None

        self.q_evolution = []
        self.nll = 0.0
        self.reversal_trials = []

        for t in range(total_trials):
            state = str(self.states[t])
            probs = self._softmax_policy(q_table, state)

            # choose action
            if t < len(self.past_states):
                action = str(self.past_actions[t]).upper()
            else:
                action = random.choices(self.action_space, weights=probs)[0]
                self.actions[t] = action

            # accumulate NLL
            self._a2i = {a:i for i,a in enumerate(self.action_space)}
            act_idx = self._a2i[action]
            self.nll += -np.log(probs[act_idx] + 1e-12)

            # reward + Q update
            reward = self._reward_gen(action, state)
            self.Rewards[t] = int(reward)
            q_table[(action, state)] = (1 - self.a) * q_table[(action, state)] + self.a * reward

            # track Q snapshots
            self.q_evolution.append(q_table.copy())

            # performance tracking (correctness)
            is_corr = 1 if self.correct_pairs.get((state, action), 0) == 1 else 0
            last_k_correct.append(is_corr)
            if len(last_k_correct) > self.perf_window:
                last_k_correct.pop(0)

            # possibly trigger reversal window
            if (not reversal_triggered
                and len(last_k_correct) == self.perf_window
                and sum(last_k_correct) >= self.perf_threshold):
                print(f"\n🔔 Rule reversal trigger at trial {t+1} (≥{self.perf_threshold}/{self.perf_window} correct).")
                self.reversal_trials.append(t+1)
                reversal_triggered = True
                countdown = self.reversal_countdown_init

            # countdown to reversal and flip when it hits 0
            if reversal_triggered:
                countdown -= 1
                if countdown == 0:
                    self._reverse_rewards()
                    reversal_triggered = False  # allow future reversals if criterion re-satisfied

            # counts
            self.state_counts[state] += 1
            self.action_counts[action] += 1
            self.state_action_counts[(state, action)] += 1

        self.last_qs = q_table
        return self.states, self.actions, self.Rewards, self.reversal_trials, [int(x) for x in last_k_correct]

    def generateOptT(self):
        total_trials = len(self.past_states) + self.trials

        self.states  = self.past_states + [random.choice(self.state_space) for _ in range(self.trials)]
        self.actions = self.past_actions + [None] * self.trials
        self.Rewards = self.past_rewards + [None] * self.trials

        q_table = self._initialize_q_table(previous_q=None)
        last_k_correct = []
        # reversal disabled: never arm/flip
        reversal_triggered = False
        countdown = None

        self.q_evolution = []
        self.nll = 0.0
        self.reversal_trials = []  # keep empty to avoid phase splits downstream

        for t in range(total_trials):
            state = str(self.states[t])
            probs = self._softmax_policy(q_table, state)

            if t < len(self.past_states):
                action = str(self.past_actions[t]).upper()
            else:
                action = random.choices(self.action_space, weights=probs)[0]
                self.actions[t] = action

            act_idx = self.action_space.index(action)
            self.nll += -np.log(probs[act_idx] + 1e-12)

            reward = self._reward_gen(action, state)
            self.Rewards[t] = int(reward)
            q_table[(action, state)] = (1 - self.a) * q_table[(action, state)] + self.a * reward

            self.q_evolution.append(q_table.copy())

            # correctness window still computed (handy for diagnostics), but no trigger/flip
            is_corr = 1 if self.correct_pairs.get((state, action), 0) == 1 else 0
            last_k_correct.append(is_corr)
            if len(last_k_correct) > self.perf_window:
                last_k_correct.pop(0)

            # ---- Disabled trigger & countdown ----
            # if (len(last_k_correct) == self.perf_window and sum(last_k_correct) >= self.perf_threshold):
            #     # previously: print, append trigger, arm countdown...
            #     pass
            # if reversal_triggered:
            #     countdown -= 1
            #     if countdown == 0:
            #         self._reverse_rewards()
            #         reversal_triggered = False
            # --------------------------------------

            self.state_counts[state] += 1
            self.action_counts[action] += 1
            self.state_action_counts[(state, action)] += 1

        self.last_qs = q_table
        return self.states, self.actions, self.Rewards, self.reversal_trials, [int(x) for x in last_k_correct]


class SyntheticDATA_Generator_dynamic_rev:
    def __init__(self, max_trials, state_space, action_space,
                 reward_probs, correct_pairs, schedule_choice):
        self.max_trials = max_trials
        self.state_space = state_space
        self.action_space = action_space
        self.reward_probs = reward_probs.copy()
        self.correct_pairs = correct_pairs.copy()
        self.schedule_choice = schedule_choice  # "exponential" or "linear"

        self.nll = 0
        self.last_qs = None
        self.rewards = defaultdict(list)

    def alpha_schedule(self, t):
        if self.schedule_choice == "exponential":
            return np.exp(-t / 100)  # divide by 100 to avoid alpha going to 0 too fast
        elif self.schedule_choice == "linear":
            return max(0, 1 - t / self.max_trials)

    def beta_schedule(self, t):
        if self.schedule_choice == "exponential":
            return 1 / (1 + np.exp(-t / 100))  # divide by 100 to slow the growth
        elif self.schedule_choice == "linear":
            return min(10, t / (self.max_trials / 2))  # cap at beta=10
        
    def initialize_q_table(self):
        return {(action, state): 0.0 for action in self.action_space for state in self.state_space}

    def reward_gen(self, action, state):
        r = random.uniform(0, 1)
        prob = self.reward_probs.get((state, action), 0)
        reward = 1 if r < prob else 0
        self.rewards[(action, state)].append(reward)
        return reward

    def softmax_policy(self, q_table, state, beta):
        qs = np.array([q_table[(action, state)] for action in self.action_space])
        exp_qs = np.exp(beta * qs)
        probs = exp_qs / (np.sum(exp_qs) + 1e-5)
        return probs

    def reverse_rewards(self):
        """Reverse reward probabilities: correct pairs become incorrect and vice versa."""
        new_reward_probs = {}
        new_correct = {}

        for (state, action) in self.reward_probs:
            old_prob = self.reward_probs[(state, action)]
            new_prob = 1 - old_prob  # Reverse

            if action not in ['N', 'n']:
                new_reward_probs[(state, action)] = new_prob
                if new_prob > 0.5:
                    new_correct[(state, action)] = 1
                else:
                    new_correct[(state, action)] = 0
            else:
                new_reward_probs[(state, action)] = 0
                new_correct[(state, action)] = 0


        self.reward_probs = new_reward_probs
        self.correct_pairs = new_correct
        print("\n🔄 Rule reversal applied!")


    def generate(self):
        q_table = self.initialize_q_table()
        states = []
        actions = []
        rewards = []
        q_evolution = []
        self.reversal_trials = []
        last_20_correct = []
        reversal_triggered = False
        reversal_countdown = None

        for t in range(self.max_trials):
            state = random.choice(self.state_space)

            alpha = self.alpha_schedule(t)
            beta = self.beta_schedule(t)
            probs = self.softmax_policy(q_table, state, beta)
            action = random.choices(self.action_space, weights=probs)[0]

            reward = self.reward_gen(action, state)
            
            q_table[(action, state)] += alpha * (reward - q_table[(action, state)])

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            q_evolution.append(q_table.copy())

            # --- Track correctness ---
            is_correct = 1 if (state, action) in self.correct_pairs else 0
            last_20_correct.append(is_correct)
            if len(last_20_correct) > 20:
                last_20_correct.pop(0)

            # --- Reversal logic ---
            if not reversal_triggered and len(last_20_correct) == 20 and sum(last_20_correct) >= 19:
                print(f"\n🔔 19 or 20 correct out of 20 at trial {t + 1}. Starting 250 trial countdown.")
                self.reversal_trials.append(t + 1)
                reversal_triggered = True
                reversal_countdown = 250

            if reversal_triggered:
                reversal_countdown -= 1
                if reversal_countdown == 0:
                    choice = input("Continue to reversal or end? (yes/no): ").strip().lower()
                    if choice == "yes":
                        self.reverse_rewards()
                        reversal_triggered = False
                    else:
                        print("\n✅ User chose to end data generation early.")
                        break  # End the simulation early

        self.q_evolution = q_evolution
        return states, actions, rewards, self.reversal_trials


# ----------------------------------------
# SYNTHETIC DATA GENERATOR (VARIABLE + CONSTANT MODEL)
# ----------------------------------------

class SyntheticDATA_Generator_constantModel_var:
    def __init__(self, a, b, max_trials, state_space, action_space,
             reward_probs, correct_pairs):
        self.a = a
        self.b = b
        self.max_trials = max_trials  # <<< new top limit
        self.state_space = state_space
        self.action_space = action_space
        self.reward_probs = reward_probs.copy()
        self.correct_pairs = correct_pairs.copy()


        self.nll = 0
        self.last_qs = None
        self.rewards = defaultdict(list)

    def initialize_q_table(self):
        return {(action, state): 0.0 for action in self.action_space for state in self.state_space}

    def reward_gen(self, action, state):
        r = random.uniform(0, 1)
        prob = self.reward_probs.get((state, action), 0)
        reward = 1 if r < prob else 0
        self.rewards[(action, state)].append(reward)
        return reward

    def softmax_policy(self, q_table, state):
        qs = np.array([q_table[(action, state)] for action in self.action_space])
        exp_qs = np.exp(self.b * qs)
        probs = exp_qs / (np.sum(exp_qs) + 1e-5)
        return probs

    def get_user_updated_reward_probs(self):
        print("\n🔄 Enter new reward probabilities for each (state, action):")
        new_probs = {}
        new_correct = {}
        for state in self.state_space:
            for action in self.action_space:
                while True:
                    try:
                        val = float(input(f"P(reward | state={state}, action={action}): ").strip())
                        if 0 <= val <= 1:
                            new_probs[(state, action)] = val
                            break
                        else:
                            print("Enter value between 0 and 1.")
                    except ValueError:
                        print("Invalid number.")
        return new_probs, new_correct

    def generate(self):
        q_table = self.initialize_q_table()
        states = []
        actions = []
        rewards = []
        q_evolution = []
        last_20_correct = []
        reversal_triggered = False
        reversal_countdown = None
        total_trials = 0
        checked_19of20_once = False

        keep_running = True

        # --------- FIRST PHASE: run until 19/20 correct or max_trials ---------
        while keep_running and total_trials < self.max_trials:
            state = random.choice(self.state_space)
            probs = self.softmax_policy(q_table, state)
            action = random.choices(self.action_space, weights=probs)[0]

            reward = self.reward_gen(action, state)
            q_table[(action, state)] += self.a * (reward - q_table[(action, state)])

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            q_evolution.append(q_table.copy())

            total_trials += 1

            is_correct = 1 if (state, action) in self.correct_pairs else 0
            last_20_correct.append(is_correct)
            if len(last_20_correct) > 20:
                last_20_correct.pop(0)

            # Trigger only once
            if (not checked_19of20_once and not reversal_triggered and
                len(last_20_correct) == 20 and sum(last_20_correct) >= 19):
                print(f"\n🔔 19 or 20 correct out of 20 at trial {total_trials}. Starting 500 trial countdown.")
                reversal_triggered = True
                reversal_countdown = 500
                checked_19of20_once = True

            # Handle countdown
            if reversal_triggered:
                reversal_countdown -= 1
                if reversal_countdown == 0:
                    print(f"\n🕒 500 trials passed since trigger. Asking for new reward probabilities.")
                    choice = input("Provide new reward probabilities or end? (new/end): ").strip().lower()
                    if choice == "new":
                        new_probs, new_correct = self.get_user_updated_reward_probs()
                        self.reward_probs = new_probs
                        self.correct_pairs = new_correct
                        last_20_correct = []
                        break  # Move to phase 2
                    else:
                        keep_running = False
                        break

        # --------- PHASE 2+: 500-trial blocks, no more 19/20 checking ---------
        while keep_running and total_trials < self.max_trials:
            print("\n🔄 Starting a new block of 500 trials.")

            block_trials = 0

            while block_trials < 500 and total_trials < self.max_trials:
                state = random.choice(self.state_space)
                probs = self.softmax_policy(q_table, state)
                action = random.choices(self.action_space, weights=probs)[0]

                reward = self.reward_gen(action, state)
                q_table[(action, state)] += self.a * (reward - q_table[(action, state)])

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                q_evolution.append(q_table.copy())

                block_trials += 1
                total_trials += 1

            # After each block, ask the user
            if total_trials < self.max_trials:
                print(f"\n🕒 500 trials completed. Total trials so far: {total_trials}.")
                choice = input("Provide new reward probabilities or end? (new/end): ").strip().lower()
                if choice == "new":
                    new_probs, new_correct = self.get_user_updated_reward_probs()
                    self.reward_probs = new_probs
                    self.correct_pairs = new_correct
                else:
                    keep_running = False

        self.q_evolution = q_evolution
        return states, actions, rewards
    
# ----------------------------------------
# Estimating Model Parameters over Windows of trials in rev Protocol
# ----------------------------------------
from scipy.optimize import minimize
import numpy as np

import numpy as np
from scipy.optimize import minimize

class ModelParameterEstimatesFromSAR_rev:
    """
    Windowed MLE of (alpha, beta, bias_L, bias_R, bias_N) for a softmax Q-learning policy.
    Optimized: uses NumPy Q-table + precomputed state/action indices.
    """

    def __init__(self, realOrSim, l, w, full_states, full_actions, full_rewards,
                 reversal_trials, mouse_type=None, protocol=None, data_loader=None):

        self.l = int(l)
        self.w = int(w)
        self.realOrSim = str(realOrSim)
        self.protocol = protocol
        self.mouse_type = mouse_type
        self.data_loader = data_loader
        self.reversal_trials = list(reversal_trials or [])

        # ---- Auto-load (unchanged feature) ----
        if (self.realOrSim.lower() == "real"
            and self.data_loader is not None
            and (not full_states or not full_actions or not full_rewards)):
            df = self.data_loader.load_data()

            state_col  = 'tone_freq' if 'tone_freq' in df.columns else ('state' if 'state' in df.columns else None)
            action_col = 'response'  if 'response'  in df.columns else ('action' if 'action' in df.columns else None)
            reward_col = 'rew_t'     if 'rew_t'     in df.columns else ('reward' if 'reward' in df.columns else None)

            if not all([state_col, action_col, reward_col]):
                missing = [name for name, ok in [
                    ('tone_freq/state', state_col),
                    ('response/action', action_col),
                    ('rew_t/reward', reward_col)
                ] if ok is None]
                raise ValueError(f"Could not infer columns. Missing: {', '.join(missing)}")

            full_states  = df[state_col].astype(str).tolist()
            full_actions = df[action_col].astype(str).str.upper().tolist()
            full_rewards = df[reward_col].astype(int).tolist()

        # normalize
        self.full_states  = [str(s) for s in full_states]
        self.full_actions = [str(a).upper() for a in full_actions]
        self.full_rewards = [int(r) for r in full_rewards]

        # fixed action set in deterministic order (unchanged)
        self.ActionSpace = ['L', 'R', 'N']
        self.action_to_i = {'L': 0, 'R': 1, 'N': 2}

        # state space inferred from data
        self.StateSpace = sorted(set(self.full_states))
        if not self.StateSpace:
            raise ValueError("Empty StateSpace inferred from data.")
        self.state_to_i = {s: i for i, s in enumerate(self.StateSpace)}

        # cached scratch arrays to reduce allocations
        self._u = np.zeros(3, dtype=float)
        self._p = np.zeros(3, dtype=float)

        # phase bounds
        self.bottom_rung = 0
        self.top_rung = len(self.full_states) - 1

        # window scratch (views)
        self._ST = []
        self._AC = []
        self._RW = []

    # ---------- phase logic (unchanged behavior) ----------
    def PHASES_SPLIT_BY_REVERSAL(self):
        if self.realOrSim.lower() == 'sim':
            phases = []
            final_expertness = 1 if self.reversal_trials else 0
            if not self.reversal_trials:
                phases.append([0, len(self.full_states) - 1])
            else:
                start = 0
                for c in self.reversal_trials:
                    c = int(c)
                    phases.append([start, max(c - 1, start)])
                    start = c
                phases.append([start, len(self.full_states) - 1])
        elif self.realOrSim.lower() == 'real':
            if self.data_loader is None:
                raise ValueError("data_loader must be provided for real data.")
            phases, final_expertness, _ = self.data_loader.get_learning_phases()
        else:
            raise ValueError("realOrSim must be 'real' or 'sim'.")
        return phases, final_expertness

    # ---------- data slicing (faster: store indices + local refs) ----------
    def DATA(self, tau):
        start = self.bottom_rung + self.w * tau
        end   = min(start + self.l, self.top_rung + 1)
        self._ST = self.full_states[start:end]
        self._AC = self.full_actions[start:end]
        self._RW = self.full_rewards[start:end]
        if not self._ST:
            raise ValueError("Empty window slice; check l, w, and phase bounds.")

    # ---------- fast NLL using NumPy Q-table ----------
    def SoftMaxGivenData(self, q_i, alpha, beta, biasL, biasR):
        """
        Same model:
          bN = -(bL + bR) (zero-sum)
          softmax over [L,R,N] of beta*Q + bias
          Q update: Q <- (1-a)Q + a*r for chosen (a,s)
        """
        alpha = float(alpha); beta = float(beta)
        bL = float(biasL); bR = float(biasR); bN = -(bL + bR)

        # biases aligned to ActionSpace order: [L,R,N]
        b = np.array([bL, bR, bN], dtype=float)

        # local copy so objective doesn't mutate caller state
        Q = q_i.copy()

        nll = 0.0
        eps = 1e-12

        u = self._u
        p = self._p

        for s, a, r in zip(self._ST, self._AC, self._RW):
            si = self.state_to_i[s]
            ai = self.action_to_i.get(a, None)
            if ai is None:
                raise ValueError(f"Unexpected action '{a}'. Expected one of {self.ActionSpace}.")

            # u = beta*Q[:,si] + b
            np.multiply(Q[:, si], beta, out=u)
            u += b

            # stable softmax
            m = u.max()
            u -= m
            np.exp(u, out=p)
            Z = p.sum() + eps
            p /= Z

            nll += -np.log(p[ai] + eps)

            # Q update
            Q[ai, si] = (1.0 - alpha) * Q[ai, si] + alpha * float(r)

        return nll

    # ---------- propagate Q to next window start (fast) ----------
    def SoftMaxGivenDataWW(self, tau, q_in, opt_alpha):
        slice_start = self.bottom_rung + self.w * tau
        slice_end   = min(self.bottom_rung + self.w * (tau + 1), self.top_rung + 1)

        ST = self.full_states[slice_start:slice_end]
        AC = self.full_actions[slice_start:slice_end]
        RW = self.full_rewards[slice_start:slice_end]

        Q = q_in.copy()
        a = float(opt_alpha)

        for s, act, r in zip(ST, AC, RW):
            si = self.state_to_i[s]
            ai = self.action_to_i.get(act, None)
            if ai is None:
                raise ValueError(f"Unexpected action '{act}'. Expected one of {self.ActionSpace}.")
            Q[ai, si] = (1.0 - a) * Q[ai, si] + a * float(r)

        return Q

    # ---------- main estimator ----------
    def estimator(self):
        ALPHAS, BETAS, BiasL, BiasR, BiasN, COLORS = [], [], [], [], [], []
        phases, final_expertness = self.PHASES_SPLIT_BY_REVERSAL()

        # Q-start as NumPy table (3 x |S|)
        q_start = np.zeros((3, len(self.StateSpace)), dtype=float)

        for phase in phases:
            self.bottom_rung = int(phase[0])
            self.top_rung    = int(phase[1])
            phase_len = self.top_rung - self.bottom_rung + 1

            # robust window count (same behavior)
            if phase_len <= self.l:
                num_windows = 1
            else:
                num_windows = 1 + max(0, (phase_len - self.l) // max(1, self.w))

            q_curr = q_start

            for tau in range(num_windows):
                self.DATA(tau)

                if tau == 0 and not ALPHAS:
                    x0 = np.array([0.5, 1.0, 0.0, 0.0], dtype=float)
                else:
                    x0 = np.array([ALPHAS[-1], BETAS[-1], BiasL[-1], BiasR[-1]], dtype=float)

                # objective closes over current q_curr and current window data
                def obj(params):
                    return self.SoftMaxGivenData(q_curr, params[0], params[1], params[2], params[3])

                result = minimize(
                    obj,
                    x0=x0,
                    bounds=[(0.0, 1.0), (0.0, 50.0), (-20.0, 20.0), (-20.0, 20.0)],
                    method='Powell'
                )

                a_hat, b_hat, bL_hat, bR_hat = result.x
                bN_hat = -(bL_hat + bR_hat)

                ALPHAS.append(float(a_hat))
                BETAS.append(float(b_hat))
                BiasL.append(float(bL_hat))
                BiasR.append(float(bR_hat))
                BiasN.append(float(bN_hat))

                # color logic preserved
                if tau == num_windows - 1:
                    if phase == phases[-1]:
                        COLORS.append('blue' if final_expertness == 1 else 'black')
                    else:
                        COLORS.append('black')
                else:
                    COLORS.append('blue')

                # carry Q forward (same “memory” feature)
                q_curr = self.SoftMaxGivenDataWW(tau, q_curr, a_hat)

            q_start = q_curr

        return ALPHAS, BETAS, BiasL, BiasR, BiasN, COLORS


    # ---------- (unchanged) Q reconstruction from fixed params ----------
    def estimate_q_evolution(self, alphas, betas, bias1, bias2, bias3):
        q_evolution = []
        q_start = {(a, s): 0.0 for a in self.ActionSpace for s in self.StateSpace}
        memory = 0
        param_index = 0
        phases, _ = self.PHASES_SPLIT_BY_REVERSAL()

        for phase in phases:
            self.bottom_rung = phase[0]
            self.top_rung = phase[1]

            if (self.top_rung - self.bottom_rung + 1 <= self.l):
                num_windows = 1
            else:
                num_windows = (self.top_rung + 1 - self.l - self.bottom_rung) // self.w

            for tau in range(num_windows):
                if param_index >= len(alphas) or param_index >= len(betas):
                    raise IndexError(f"param_index {param_index} exceeds available parameters: alphas={len(alphas)}, betas={len(betas)}")

                a_hat = alphas[param_index]
                b_hat = betas[param_index]
                Bias1_hat = bias1[param_index]
                Bias2_hat = bias2[param_index]
                Bias3_hat = bias3[param_index]
                param_index += 1

                start = self.bottom_rung + tau * self.w
                end = start + self.l if tau < num_windows - 1 else self.top_rung + 1

                self.STATES = self.full_states[start:end]
                self.ACTIONS = self.full_actions[start:end]
                self.REWARDS = self.full_rewards[start:end]

                self.initialize_q_table(memory, q_start)

                for t in range(len(self.STATES)):
                    state = self.STATES[t]
                    action = self.ACTIONS[t]
                    reward = self.REWARDS[t]
                    self.Q[(action, state)] += a_hat * (reward - self.Q[(action, state)])

                q_evolution.append(self.Q.copy())

            q_start = self.Q.copy()
            memory = 1

        if param_index != len(alphas):
            raise ValueError(f"Used {param_index} alpha-beta pairs, but {len(alphas)} were generated.")

        return q_evolution
