import random
#Differential evolution optimizer based on Scipy implementation:
#http://python-scipy.sourcearchive.com/documentation/0.6.0/classscipy_1_1sandbox_1_1rkern_1_1diffev_1_1DiffEvolver.html

def argmin(x):
    if len(x) < 2:
        return x[0]
    bestv,idx = x[0],0
    for e,i in enumerate(x[1:],1):
        if i<bestv:
            bestv = i
            idx = e
    return idx

class DiffEvolver(object):
    """Minimize a function using differential evolution.

    Constructors
    ------------
    DiffEvolver(func, pop0, args=(), crossover_rate=0.5, scale=None,
        strategy=('rand', 2, 'bin'), eps=1e-6)
      func -- function to minimize
      pop0 -- sequence of initial vectors
      args -- additional arguments to apply to func
      crossover_rate -- crossover probability [0..1] usually 0.5 or so
      scale -- scaling factor to apply to differences [0..1] usually > 0.5
        if None, then calculated from pop0 using a heuristic
      strategy -- tuple specifying the differencing/crossover strategy
        The first element is one of 'rand', 'best', 'rand-to-best' to specify
        how to obtain an initial trial vector.
        The second element is either 1 or 2 (or only 1 for 'rand-to-best') to
        specify the number of difference vectors to add to the initial trial.
        The third element is (currently) 'bin' to specify binomial crossover.
      eps -- if the maximum and minimum function values of a given generation are
        with eps of each other, convergence has been achieved.

    DiffEvolver.frombounds(func, lbound, ubound, npop, crossover_rate=0.5,
        scale=None, strategy=('rand', 2, 'bin'), eps=1e-6)
      Randomly initialize the population within given rectangular bounds.
      lbound -- lower bound vector
      ubound -- upper bound vector
      npop -- size of population

    Public Methods
    --------------
    solve(newgens=100)
      Run the minimizer for newgens more generations. Return the best parameter
      vector from the whole run.

    Public Members
    --------------
    best_value -- lowest function value in the history
    best_vector -- minimizing vector
    best_val_history -- list of best_value's for each generation
    best_vec_history -- list of best_vector's for each generation
    population -- current population
    pop_values -- respective function values for each of the current population
    generations -- number of generations already computed
    func, args, crossover_rate, scale, strategy, eps -- from constructor
    """

    def __init__(self, func, pop0, args=(), crossover_rate=0.5, scale=None,
            strategy=('rand', 2, 'bin'), eps=1e-6, lbound=None, ubound=None):
        self.func = func
        self.population = pop0
        self.npop, self.ndim = len(self.population),len(self.population[0])
        self.args = args
        self.crossover_rate = crossover_rate
        self.strategy = strategy
        self.eps = eps
        self.lbound = lbound
        self.ubound = ubound
        self.bounds = lbound!=None and ubound!=None

        self.pop_values = [self.func(m, *args) for m in self.population]
        bestidx = argmin(self.pop_values)
        self.best_vector = self.population[bestidx]
        self.best_value = self.pop_values[bestidx]

        if scale is None:
            self.scale = self.calculate_scale()
        else:
            self.scale = scale

        self.generations = 0
        self.best_val_history = []
        self.best_vec_history = []

        self.jump_table = {
            ('rand', 1, 'bin'): (self.choose_rand, self.diff1, self.bin_crossover),
            ('rand', 2, 'bin'): (self.choose_rand, self.diff2, self.bin_crossover),
            ('best', 1, 'bin'): (self.choose_best, self.diff1, self.bin_crossover),
            ('best', 2, 'bin'): (self.choose_best, self.diff2, self.bin_crossover),
            ('rand-to-best', 1, 'bin'):
                (self.choose_rand_to_best, self.diff1, self.bin_crossover),
            }

    def clear(self):
        self.best_val_history = []
        self.best_vec_history = []
        self.generations = 0
        self.pop_values = [self.func(m, *self.args) for m in self.population]

    def frombounds(cls, func, lbound, ubound, npop, crossover_rate=0.5,
            scale=None, x0=None, strategy=('rand', 2, 'bin'), eps=1e-6):
        if x0==None:
            pop0 = [[random.random()*(ubound[i]-lbound[i]) + lbound[i] for i in xrange(len(lbound))] for c in xrange(npop)]
        else:
            pop0 = [0]*npop
            for e,x in enumerate(x0):
                if len(x)!=len(lbound):
                    raise ValueError("Dimension of x0[{}] is incorrect".format(e))
                if any(not lbound[i]<=x[i]<=ubound[i] for i in xrange(len(lbound))):
                    raise ValueError("x0[{}] not inside the bounds.".format(e))
                for i in xrange(e,npop,len(x0)):
                    pop0[i] = x
            delta = 0.3
            pop0 = [[delta*(random.random()*(ubound[i]-lbound[i]) + lbound[i])+p[i] for i in xrange(len(lbound))] for p in pop0]
            pop0 = [[lbound[i] if p[i]<lbound[i] else (ubound[i] if p[i]>ubound[i] else p[i]) for i in xrange(len(lbound))] for p in pop0]
            #Make sure to include x0
            pop0[:len(x0)] = x0
        return cls(func, pop0, crossover_rate=crossover_rate, scale=scale,
            strategy=strategy, eps=eps, lbound=lbound, ubound=ubound)
    frombounds = classmethod(frombounds)

    def calculate_scale(self):
        rat = abs(max(self.pop_values)/self.best_value)
        rat = min(rat, 1./rat)
        return max(0.3, 1.-rat)

    def bin_crossover(self, oldgene, newgene):
        new = oldgene[:]
        for i in xrange(len(oldgene)):
            if random.random() < self.crossover_rate:
                new[i] = newgene[i]
        return new

    def select_samples(self, candidate, nsamples):
        possibilities = range(self.npop)
        possibilities.remove(candidate)
        random.shuffle(possibilities)
        return possibilities[:nsamples]

    def diff1(self, candidate):
        i1, i2 = self.select_samples(candidate, 2)
        y = [(self.population[i1][c] - self.population[i2][c]) for c in xrange(self.ndim)]
        y = [self.scale*i for i in y]
        return y

    def diff2(self, candidate):
        i1, i2, i3, i4 = self.select_samples(candidate, 4)
        y = ([(self.population[i1][c] - self.population[i2][c]+self.population[i3][c] - self.population[i4][c]) for c in xrange(self.ndim)])
        y = [self.scale*i for i in y]
        return y

    def choose_best(self, candidate):
        return self.best_vector

    def choose_rand(self, candidate):
        i = self.select_samples(candidate, 1)[0]
        return self.population[i]

    def choose_rand_to_best(self, candidate):
        return ((1-self.scale) * self.population[candidate] +
                self.scale * self.best_vector)

    def get_trial(self, candidate):
        chooser, differ, crosser = self.jump_table[self.strategy]
        chosen = chooser(candidate)
        diffed = differ(candidate)
        new = [chosen[i] + diffed[i] for i in xrange(self.ndim)]
        trial = crosser(self.population[candidate],new)
        if self.bounds:
            if random.random() < 0.2:
                trial = self.hug_bounds(trial)
            else:
                trial = self.mirror_bounds(trial)
        return trial

    def mirror_bounds(self,trial):
        """Mirrors values over bounds back to bounded area,
        or randomly generates a new coordinate if mirroring failed."""
        for i in xrange(self.ndim):
            if trial[i]<self.lbound[i]:
                trial[i] = 2*self.lbound[i]-trial[i]
                if trial[i]<self.lbound[i]:
                    trial[i] = random.random()*(self.ubound[i]-self.lbound[i]) + self.lbound[i]
            elif trial[i]>self.ubound[i]:
                trial[i] = 2*self.ubound[i]-trial[i]
                if trial[i]>self.ubound[i]:
                    trial[i] = random.random()*(self.ubound[i]-self.lbound[i]) + self.lbound[i]
        return trial

    def hug_bounds(self,trial):
        """Rounds values over bounds to bounds"""
        for i in xrange(self.ndim):
            if trial[i]<self.lbound[i]:
                trial[i] = self.lbound[i]
            elif trial[i]>self.ubound[i]:
                trial[i] = self.ubound[i]
        return trial

    def converged(self):
        return max(self.pop_values) - min(self.pop_values) <= self.eps

    def solve(self, newgens=100):
        """Run for newgens more generations.

        Return best parameter vector from the entire run.
        """
        for gen in xrange(self.generations+1, self.generations+newgens+1):
            for candidate in range(self.npop):
                trial = self.get_trial(candidate)
                trial_value = self.func(trial, *self.args)
                if trial_value < self.pop_values[candidate]:
                    self.population[candidate] = trial
                    self.pop_values[candidate] = trial_value
                    if trial_value < self.best_value:
                        self.best_vector = trial
                        self.best_value = trial_value
            self.best_val_history.append(self.best_value)
            self.best_vec_history.append(self.best_vector)
            if self.converged():
                break
        self.generations = gen
        return self.best_vector
