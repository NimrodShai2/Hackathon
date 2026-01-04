import numpy as np


class HMM:
    def __init__(self, emissions, transitions):
        self.emissions = emissions  # Emission probability matrix
        self.transitions = transitions  # Transition probability matrix
        self.pi = np.array([0.8, 0.2, 0, 0])  # Initial state distribution (pi)

    def run_viterbi(self):
        n_states = self.transitions.shape[0]
        n_observations = self.emissions.shape[1]

        forward_probs = np.zeros((n_states, n_observations))
        backtracking = np.zeros((n_states, n_observations), dtype=int)

        # initialize base cases (t == 0)
        for s in range(n_states):
            forward_probs[s, 0] = self.pi[s] * self.emissions[s, 0]

        #recursion step
        for o in range(1, n_observations):
            for s in range(n_states):
                transition_probs = forward_probs[:, o - 1] * self.transitions[:, s]
                best_prev_state = np.argmax(transition_probs)
                forward_probs[s, o] = transition_probs[best_prev_state] * self.emissions[s, o]
                backtracking[s, o] = best_prev_state

        #termination step
        states = np.zeros(n_observations, dtype=int)
        states[-1] = np.argmax(forward_probs[:, n_observations - 1])

        #backtracking step
        for o in range(n_observations - 2, -1, -1):
            states[o] = backtracking[states[o + 1], o + 1]

        return states
