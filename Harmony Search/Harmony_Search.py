import numpy as np
import random


# def objective_function(x):
#     return x[0] ** 2 + x[1] ** 2

# def objective_function(x):
#     return np.log(1 + (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

# def objective_function(x):
#     return -np.sin(x[0]) * (np.sin((x[0] ** 2)/np.pi)) ** 20 - np.sin(x[1]) * (np.sin((2 * x[1] ** 2)/np.pi)) ** 20

d = 2
def objective_function(x):
    return 418.9829 * d - sum([x[i] * np.sin(np.sqrt(abs(x[i]))) for i in range(d)])

class Harmony_Search:
    def __init__(self, objective, restrictions=None, HMCR=0.7, HMS=10,
                 MI=5000, FW_min=0.005, FW_max=0.5, PAR_min=0.5, PAR_max=0.95):
        self.objective = objective
        self.restrictions = restrictions
        self.HMCR = HMCR
        self.HMS = HMS
        self.MI = MI
        self.FW_min = FW_min
        self.FW_max = FW_max
        self.PAR_min = PAR_min
        self.PAR_max = PAR_max
        self.x_span = np.linspace(0, np.pi / FW_max)
        self.y_span = np.linspace(0, np.pi / FW_max)
        self.bounds = np.array([self.x_span, self.y_span])
        self.number_of_variables = self.bounds.shape[0]
        self.memory = self.initialize_memory()

    def boundaries(self, t):
        if self.FW(t) < self.FW_min:
            fw = self.FW_min
        else:
            fw = self.FW(t)
        x_span = np.linspace(0, np.pi / fw)
        y_span = np.linspace(0, np.pi / fw)
        boundaries = np.array([x_span, y_span])
        return boundaries

    def Sort(self, s):
        return sorted(s, key=lambda z: z[1])

    def initialize_memory(self):
        vectors = []
        values = np.zeros(self.HMS)
        HM = []
        for i in range(self.HMS):

            v = np.random.uniform(0, np.pi, size=self.number_of_variables)
            vectors.append(v)
            value = self.objective(v)
            values[i] += value
        evaluation = self.Sort(list(zip(vectors, values)))
        for i in range(self.HMS):
            HM.append(evaluation[i][0])
        return HM

    def PAR(self, t):
        par = self.PAR_min + (self.PAR_max - self.PAR_min) * (t / self.MI)
        return par

    def FW(self, t):
        fw = self.FW_max * ((self.FW_min / self.FW_max) ** (t / self.MI))
        return fw

    def searching_algorithm(self):
        t = 0
        while t < self.MI:
            intervals = self.boundaries(t)
            prob = np.random.uniform(0, 1)
            if prob < 1 - self.HMCR:
                x_new = []
                for m in range(self.number_of_variables):
                    x_new.append(np.random.uniform(low=intervals[m][0], high=intervals[m][-1]))

                for i in reversed(range(self.HMS)):
                    if self.objective(x_new) <= self.objective(self.memory[i]):
                        self.memory[i] = x_new
                        break
            elif prob > 1 - self.HMCR and prob < 1 - self.HMCR + self.HMCR * (1 - self.PAR(t)):
                pass
            elif prob > 1 - self.HMCR + self.HMCR * (1 - self.PAR(t)):
                x_new = random.choice(self.memory)
                for m in range(self.number_of_variables):
                    x_new[m] += np.random.uniform(0, 1) * self.FW(t)
                for i in reversed(range(self.HMS)):
                    if self.objective(x_new) <= self.objective(self.memory[i]):
                        self.memory[i] = x_new
                        break
            t += 1
            print('Epoch: {} --- State: {} --- Objective: {:.2f}'.format(t, np.round(self.memory[-1], 2), self.objective(self.memory[-1])))

a = Harmony_Search(objective=objective_function, HMS=30, MI=40000)
a.searching_algorithm()

