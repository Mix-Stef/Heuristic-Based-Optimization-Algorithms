import numpy as np

"""Implemention of Simulated Annealing algorithm for constrained or unconstrained optimization problems. It is a derivative-free algorithm and can be 
used for discrete and continuous functions."""

# def energy_function(x):
#     return x[0] ** 2 + x[0] * x[1] + x[1] ** 2 - 3 * x[0]
#
# def energy_function(x):
#     return 2 * x[0] ** 3 + 6 * x[0] * x[1] ** 2 - 3 * x[1] ** 3 - 150 * x[0]

# def energy_function(x):
#     return 2 * x[0] * x[2] + 2 * x[1] * x[2] + x[0] * x[1]


d = 4
def energy_function(x):
    return 418.9829 * d - sum([x[i] * np.sin(np.sqrt(abs(x[i]))) for i in range(d)])

#
# def restrictions(x):
#     eps = 0.001
#     g1 = x[0] ** 2 + x[1] ** 2 - 1 - eps
#     g2 = - x[0] ** 2 - x[1] ** 2 + 1 - eps
#     return g1, g2

# def energy_function(x):
#     return  (x[0] ** 2) * x[1]

def restrictions(x):
    r1 = x[0] - 500
    r2 = -x[0] - 500
    r3 = x[1] - 500
    r4 = -x[1] - 500
    r5 = x[2] - 500
    r6 = -x[2] - 500
    r7 = x[3] - 500
    r8 = -x[3] - 500
    return r1, r2, r3, r4, r5, r6, r7, r8

# def restrictions(x):
#     eps = 0.01
#     # g1 = x[0] ** 2 + x[1] ** 2 - 3 - eps
#     # g2 = - x[0] ** 2 - x[1] ** 2 + 3 - eps
#     g1 = x[0] * x[1] * x[2] - 10 - eps
#     g2 = - x[0] * x[1] * x[2] + 10 - eps
#     return g1, g2

class Simulated_Annealing:
    def __init__(self, T_max, T_min, energy, initial_point, constraints=None):
        self.T_start = T_max # Initial temperature
        self.T_end = T_min # Final temperature
        self.energy = energy # Objective function
        self.initial_point = initial_point 
        self.constraints = constraints # Optional
        self.dimension = len(self.initial_point)
        if self.constraints is not None:
            self.number_of_restrictions = len(self.constraints(self.initial_point))

    def delta_energy(self, state, next_state):
        """Returns the change of energy after a transition"""
        return self.energy(next_state) - self.energy(state)

    def transition_probability(self, de, t):
        """Returns the transition probability"""
        return np.exp(- de / t)

    def choose_next_state(self, state):
        """At each time step the exporation domain changes"""
        eps = np.random.uniform(50, 100)
        next_state = np.zeros_like(state)
        for i in range(len(state)):
            next_state[i] += np.random.uniform(state[i] - eps, state[i] + eps)
        return next_state

    def cooling(self, time_step):
        """The cooling process can be implemented as a linear or exponential decrease"""

        # T = self.T_start * np.exp(- 0.003 * time_step)
        T = self.T_start / (time_step + 1)

        return T

    def transition(self, state, temp):
        next_state = self.choose_next_state(state)
        energy_diff = self.delta_energy(state, next_state)
        prob = self.transition_probability(energy_diff, temp)
        if self.constraints is not None:
            """We check if all restrictions are satisfied"""
            if sum([self.constraints(next_state)[i] < 0 for i in range(self.number_of_restrictions)]) == self.number_of_restrictions:
                if energy_diff < 0:
                    state = next_state
                    return state
                else:
                    rand = np.random.uniform(0, 1)
                    if rand < prob:
                        state = next_state
                        return state
                    else:
                        return state
            else:
                return state
        else:
            if energy_diff < 0:
                state = next_state
                return state
            else:
                rand = np.random.uniform(0, 1)
                if rand < prob:
                    state = next_state
                    return state
                else:
                    return state

    def annealing(self):
        """Initialization"""
        state = self.initial_point
        temp = self.T_start
        t = 0
        """At each time step we decrease the temperature and perform the transition"""
        while temp > self.T_end:
            temperature = self.cooling(t + 1)
            next_state = self.transition(state, temperature)
            temp = temperature
            state = next_state
            t += 1
            # print('State {} --- Temperature {}'.format(state, temp))
        print('Done. The final equilibrium state is {}. The energy at that state is {}'.format(state, self.energy(state)))
        

if __name__ == '__main__':
    sa = Simulated_Annealing(100, 0.1, energy_function, [400, 400, 400, 400], restrictions)
    sa.annealing()

