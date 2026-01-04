import numpy as np


class HMM:
    def __init__(self, transition_matrix, emission_matrix, pi=None):
        self.transitions = np.asarray(transition_matrix, dtype=np.float64)
        self.emissions = np.asarray(emission_matrix, dtype=np.float64)
        self.n_states = self.transitions.shape[0]
        self.pi = np.asarray(pi, dtype=np.float64) if pi is not None else np.full(self.n_states, 1.0 / self.n_states, dtype=np.float64)

    def run_viterbi(self):
        # Emissions expected as shape [T, n_states]
        T = self.emissions.shape[0]
        log_trans = np.log(self.transitions + 1e-20)
        log_emit = np.log(self.emissions + 1e-20)
        log_pi = np.log(self.pi + 1e-20)

        dp = np.zeros((T, self.n_states))
        back = np.zeros((T, self.n_states), dtype=np.int32)

        dp[0] = log_pi + log_emit[0]

        for t in range(1, T):
            for s in range(self.n_states):
                scores = dp[t - 1] + log_trans[:, s]
                best_prev = np.argmax(scores)
                dp[t, s] = scores[best_prev] + log_emit[t, s]
                back[t, s] = best_prev

        states = np.zeros(T, dtype=np.int32)
        states[-1] = np.argmax(dp[-1])
        for t in range(T - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]

        return states
