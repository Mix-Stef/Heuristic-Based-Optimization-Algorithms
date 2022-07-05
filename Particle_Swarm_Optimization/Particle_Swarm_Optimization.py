import numpy as np


# def energy(x):
#     return (x[0] + 7) ** 2 + (x[1] - 1) ** 2 + 5

# def energy(x):
#     return 2 * x[0] * x[2] + 2 * x[1] * x[2] + x[0] * x[1]
#
# def restrictions(x):
#     eps = 0.0
#     g1 = x[0] * x[1] * x[2] - 13.5 - eps
#     g2 = - x[0] * x[1] * x[2] + 12.5 - eps
#     return g1, g2

d = 6
def energy(x):
    return 418.9829 * d - sum([x[i] * np.sin(np.sqrt(abs(x[i]))) for i in range(d)])

def restrictions(x):
    r1 = x[0] - 500
    r2 = -x[0] - 500
    r3 = x[1] - 500
    r4 = -x[1] - 500
    r5 = x[2] - 500
    r6 = -x[2] - 500
    r7 = x[3] - 500
    r8 = -x[3] - 500
    r9 = x[4] - 500
    r10 = -x[4] - 500
    r11 = x[5] - 500
    r12 = -x[5] - 500

    return r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12


class Particle_Swarm_Optimization:
    def __init__(self, objective, sample_point, number_of_particles, restrictions=None, c1=2, c2=2, epochs=1000):
        self.objective = objective  # Objective function
        self.sample_point = sample_point
        self.dimension = len(self.sample_point)
        self.number_of_particles = number_of_particles  # Number of particles to use
        self.restrictions = restrictions
        if self.restrictions is not None:
            self.number_of_restrictions = len(self.restrictions(self.sample_point))
        self.inertia_constant = 1
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.fitnesses = []
        self.positions = []

        """Initialization of the particles"""
        self.particles = []
        for p in range(self.number_of_particles):
            particle = []
            position = np.random.uniform(100, 200, size=(1, self.dimension))[0]
            fitness = self.objective(position)
            self.fitnesses.append(fitness)
            self.positions.append(position)
            particle.append(position)
            velocity = np.zeros_like(position)
            particle.append(velocity)
            particle.append(position)
            self.particles.append(particle)

        """We find the initial best positions"""
        self.initial_evaluation = self.Sort(list(zip(self.positions, self.fitnesses)))
        self.g_best = self.initial_evaluation[0][0]
        self.best_fitness = self.initial_evaluation[0][1]

    def Sort(self, s):
        return sorted(s, key=lambda x: x[1])

    def swarm(self):

        for t in range(1, self.epochs + 1):
            values = []
            coordinates = []
            for particle in self.particles:

                w = self.inertia_constant - (0.999 / self.epochs) * t

                new_velocity = w * particle[1] + \
                               self.c1 * (np.random.uniform(0.001, 0.999)) * (particle[2] - particle[0]) + \
                               self.c2 * (np.random.uniform(0.001, 0.999)) * (self.g_best - particle[0])
                new_position = particle[0] + new_velocity

                if self.restrictions is not None:
                    if sum([self.restrictions(new_position)[i] < 0 for i in range(self.number_of_restrictions)]) == self.number_of_restrictions:

                        particle[0] = new_position
                        particle[1] = new_velocity

                        value = self.objective(new_position)
                        values.append(value)
                        coordinates.append(new_position)

                    else:
                        value = self.objective(particle[0])
                        values.append(value)
                        coordinates.append(particle[0])

                else:
                    particle[0] = new_position
                    particle[1] = new_velocity
                    value = self.objective(new_position)
                    values.append(value)
                    coordinates.append(new_position)

            evaluation = self.Sort(list(zip(coordinates, values)))
            lowest_value = evaluation[0][1]
            best_position = evaluation[0][0]
            if lowest_value < self.best_fitness:
                self.best_fitness = lowest_value
                self.g_best = best_position

            for particle in self.particles:
                if self.objective(particle[2]) < self.objective(particle[0]):
                    pbest = particle[2]
                    particle[0] = pbest
                else:
                    pbest = particle[0]
                    particle[2] = pbest

                if self.objective(self.g_best) < self.objective(particle[0]):
                    particle[0] = self.g_best


            print('Epoch: {} --- Best Position: {} --- Objective Function: {:.2f}'.format(t, np.round(self.g_best, 2), self.objective(self.g_best)))


pso = Particle_Swarm_Optimization(energy, [0, 0, 0, 0, 0, 0], 500, c1=2, c2=2, restrictions=restrictions)
pso.swarm()
