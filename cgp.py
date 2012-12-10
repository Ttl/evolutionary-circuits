try:
    PLOTTING='matplotlib'
    import matplotlib.pyplot as plt
except:
    import subprocess
    PLOTTING='external'

import circuits
import random
from copy import deepcopy
from time import strftime,time
import re
import pickle
from os.path import join as path_join
import os
import getch
import sys
#SPICE to python circuit parser
from spice_parser import *
from math import log10
inf = 1e30
THREADS = 4

class Circuit_gene:
    """Represents a single component"""
    def __init__(self,name,nodes,cost=0,*args):
        #Name of the component(eg. "R1")
        self.spice_name = name
        #N-tuple of values
        self.values = args
        self.nodes = nodes
        self.cost = cost
    def __repr__(self):
        return self.spice_name+str(id(self))+' '+' '.join(map(str,self.nodes))+' '+' '.join(map(str,*self.values))

def log_dist(a,b):
    """Generates exponentially distributed random numbers.
    Gives better results for resistor, capacitor and inductor values
    than uniform distribution."""
    if a <= 0 or a>b:
        raise ValueError("Value out of range. Valid range is (0,infinity).")
    return 10**(random.uniform(log10(a),log10(b)))

def same(x):
    #True if all elements are same
    return reduce(lambda x,y:x==y,x)

def random_element(parts,nodes,fixed_node=None):
    #Return random circuit element from parts list
    name = random.choice(parts.keys())
    part = parts[name]
    spice_line = []
    node_list = ['n'+str(i) for i in xrange(nodes)]+['0']
    if 'value' in part.keys():
        maxval = part['max']
        minval = part['min']
        for i in xrange(part['value']):
            spice_line.append(log_dist(minval,maxval))
    nodes = [random.choice(node_list) for i in xrange(part['nodes'])]
    while same(nodes):
        nodes = [random.choice(node_list) for i in xrange(part['nodes'])]
    if fixed_node!=None:
        nodes[0]=fixed_node
        random.shuffle(nodes)
    if 'spice' in part:
        spice_line.append(part['spice'])
    if 'cost' in part:
        cost = part['cost']
    else:
        cost = 0
    return Circuit_gene(name,nodes,cost,spice_line)

def mutate_value(element,parts,rel_amount=None):
    i = random.randint(0,len(element.values)-1)
    val = element.values[i]
    name = element.spice_name
    if rel_amount==None:
        try:
            val[i] = log_dist(parts[name]['min'],parts[name]['max'])
        except:
            return element
    else:
        try:
            temp = val[i]*(2*random.random-1)*rel_amount
            if 10**parts[name]['min']<=temp<=10**parts[name]['max']*10.0:
                val[i] = temp
        except:
            return element
    try:
        cost = parts[element.spice_name]['cost']
    except KeyError:
        cost = 0
    return Circuit_gene(element.spice_name,element.nodes,cost,val)

class Chromosome:
    """Class that contains one circuit and all of it's parameters"""
    def __init__(self,max_parts,parts_list,nodes,extra_value=None):
        #Maximum number of components in circuit
        self.max_parts = max_parts
        #List of nodes
        self.nodes = nodes
        self.parts_list = parts_list
        #Generates randomly a circuit
        self.elements = [random_element(self.parts_list,self.nodes) for i in xrange(random.randint(1,max_parts/2))]
        self.extra_range = extra_value
        if extra_value!=None:
            self.extra_value = [random.uniform(*i) for i in self.extra_range]
        else:
            self.extra_value = None

    def __repr__(self):
        return str(self.elements)

    def pprint(self):
        """Pretty print function"""
        return '\n'.join(map(str,self.elements))

    def get_connected_node(self):
        """Randomly returns one connected node"""
        if len(self.elements)>0:
            device = random.choice(self.elements)
            return random.choice(device.nodes)
        else:
            return 'n1'

    def mutate(self):
        m = random.randint(0,7)
        i = random.randint(0,len(self.elements)-1)
        if m==0:
            #Change value of one component
            m = random.randint(0,1)
            if m==0:
                #New value
                self.elements[i] = mutate_value(self.elements[i],self.parts_list)
            else:
                #Slight change
                self.elements[i] = mutate_value(self.elements[i],self.parts_list,rel_amount=0.1)
        elif m==1:
            #Add one component if not already maximum number of components
            if len(self.elements)<self.max_parts:
                #self.elements.append(random_element(self.parts_list,self.nodes))
                self.elements.append(random_element(self.parts_list,self.nodes,fixed_node=self.get_connected_node()))
        elif m==2 and len(self.elements)>1:
            #Replace one component with open circuit
            del self.elements[i]
        elif m==3 and len(self.elements)>1:
            #Replace one component with open circuit
            nodes = self.elements[i].nodes
            random.shuffle(nodes)
            try:
                n1 = nodes[0]
                n2 = nodes[1]
            except IndexError:
                return None#Device doesn't have two nodes
            del self.elements[i]
            for element in self.elements:
                element.nodes = [(n1 if i==n2 else i) for i in element.nodes]
        elif m==4:
            #Replace one component keeping one node connected
            fixed_node = random.choice(self.elements[i].nodes)
            del self.elements[i]
            self.elements.append(random_element(self.parts_list,self.nodes,fixed_node=fixed_node))
        elif m==5:
            #Shuffle list of elements(better crossovers)
            random.shuffle(self.elements)
        elif m==6:
            #Change the extra_value
            if self.extra_range!=None:
                i = random.randint(0,len(self.extra_value)-1)
                self.extra_value[i] = random.uniform(*self.extra_range[i])
            else:
                self.mutate()
        elif m==7:
            #Relabel nodes
            l = len(self.elements)-1
            n1 = random.choice(self.elements[random.randint(0,l)].nodes)
            n2 = random.choice(self.elements[random.randint(0,l)].nodes)
            tries = 0
            while tries<10 or n1!=n2:
                n2 = random.choice(self.elements[random.randint(0,l)].nodes)
                tries+=1
            for element in self.elements:
                element.nodes = [(n1 if i==n2 else (n2 if i==n1 else i)) for i in element.nodes]


    def spice_input(self,options):
        """Generate the input to SPICE"""
        program = options+'\n'
        for i in self.elements:
            program+=str(i)+'\n'
        return program

    def evaluate(self,options,timeout):
        """Used in plotting, when only 1 circuits needs to be simulated"""
        program = self.spice_input(options)
        thread = circuits.spice_thread(program)
        thread.start()
        #Extra time to make sure that plotting succeeds
        thread.join(10*timeout)
        if thread.is_alive():
            try:
                thread.spice.terminate()
            except OSError:
                pass
            thread.join()
        return thread.result

def multipliers(x):
    """Convert values with si multipliers to numbers"""
    try:
        return float(x)
    except:
        pass
    try:
        a = x[-1]
        y = float(x[:-1])
        endings = {'G':9,'Meg':6,'k':3,'m':-3,'u':-6,'n':-9,'p':-12,'s':0}
        return y*(10**endings[a])
    except:
        raise ValueError("I don't know what {} means".format(x))

class CGP:
    """
    Evolutionary circuits.

    pool_size: Amount of circuits in one generation
    nodes: maximum number of available nodes, nodes are named n0,n1,n2,.. and ground node is 0(not n0)
    parts_list: List of circuit elements available
    max_parts: Integer of maximum number of circuit elements in one circuit
    elitism: Integer of number of circuits to clone driectly to next generation
    mutation_rate: Mutation propability, float in range 0.0-1.0, but not exactly 1.0, because mutation can be applied many times.
    crossover_rate: Propability of crossover.
    fitnessfunction: List of functions to test circuits against
    fitness_weight: Scores from fitness functions are weighted with this weight
    spice_sim_commands: Commands for SPICE simulator
    log: Log file to save progress
    plot_titles: Titles for the plots of cricuits
    """
    def __init__(self,
    population,
    parts,
    max_parts,
    elitism,
    mutation_rate,
    crossover_rate,
    fitness_function,
    constraints,
    spice_commands,
    log_file,
    title='',
    common='',
    models='',
    fitness_weight=None,
    constraint_weight=None,
    nodes=None,
    resumed=False,
    extra_value=None,
    plot_titles=None,
    plot_yrange=None,
    selection_weight=1,
    max_mutations=3,
    **kwargs):


        #Correct the types
        if hasattr(fitness_function, '__call__'):
            self.ff=[fitness_function]
        else:
            self.ff=fitness_function

        if hasattr(constraints, '__call__'):
            self.constraints=[constraints]
        else:
            self.constraints = constraints

        if hasattr(spice_commands, '__call__'):
            self.spice_commands=[spice_commands+common+models]
        else:
            self.spice_commands=[i+common+models for i in spice_commands]

        if fitness_weight==None:
            self.fitness_weight=[{}]
        else:
            self.fitness_weight = fitness_weight

        if self.constraints == None:
            self.constraints = [None for i in xrange(len(spice_commands))]

        if constraint_weight==None:
            self.constraint_weight=[100 for i in xrange(len(spice_commands))]
        else:
            self.constraint_weight = constraint_weight

        if len(self.spice_commands)>len(self.fitness_weight):
            raise Exception('Fitness function weight list length is incorrect')
        if len(self.spice_commands)>len(self.ff):
            raise Exception('Not enough fitness functions')
        if len(self.spice_commands)>len(self.constraints):
            raise Exception('Not enough constraints. Use None for no constraint')
        if len(self.spice_commands)>len(self.constraint_weight):
            raise Exception('Not enough constraint weights.')
        sim = map(self.parse_sim_options,self.spice_commands)

        print strftime("%Y-%m-%d %H:%M:%S")
        for e,i in enumerate(sim,1):
            print 'Simulation {0} - Type: {1}, Logarithmic plot: {2}'.format(e,i[0],str(i[1]))
            if i[3]:
                print 'Temperature specified in simulation'
                self.temperatures = True
                #TODO write the current temperature in the plot
        print

        self.timeout = kwargs['timeout']
        self.c_free_gens = kwargs['constraint_free_generations']
        self.rnd_circuits = kwargs['random_circuits']
        self.gradual_constraints = kwargs['gradual_constraints']
        self.constraint_ramp = kwargs['constraint_ramp']
        self.plot_all_gens = kwargs['plot_every_generation']

        self.def_scoring = kwargs['default_scoring']
        self.c_rank = kwargs['custom_scoring']

        if type(self.c_free_gens) not in (int,long,float):
            raise Exception('constraint_free_generations must be a number')
        if type(self.gradual_constraints) != bool:
            raise Exception('gradual_constraints must be of type "bool"')
        if type(self.constraint_ramp) not in (int,long,float):
            raise Exception('constraint_ramp must be a number')
        if type(self.plot_all_gens)!=bool:
            raise Exception('plot_every_generation must be of type "bool"')
        self.overflowed = 0

        self.cache_hits = 0
        self.cache = {}
        self.cache_size = 0
        self.cache_max_size = 1000000

        #sim_type is list of SPICE simulation types(ac,dc,tran...)
        self.sim_type = [sim[i][0] for i in xrange(len(sim))]
        #Boolean for each simulation for logarithm or linear plot
        self.log_plot = [sim[i][1] for i in xrange(len(sim))]
        #Maximum value of frequency or sweep
        self.frange   = [sim[i][2] for i in xrange(len(sim))]
        self.constraints_filled = False
        if nodes == None:
            self.nodes = max_parts
        else:
            self.nodes = nodes
        self.max_parts = max_parts
        self.extra_value = extra_value
        self.selection_weight = selection_weight

        #FIXME weight function plotting is currently disabled
        self.plot_weight=False
        #if all(i==None for i in self.fitness_weight):
        #    self.plot_weight=False
        #else:
        #    self.plot_weight=True

        if all(i==None for i in self.constraints):
            self.plot_constraints=False
        else:
            self.plot_constraints=True

        self.pool_size=population-population%THREADS+THREADS
        self.parts_list = parts
        self.generation=0
        self.elitism=elitism
        self.alltimebest=(float('inf'),float('inf'))
        self.mrate = mutation_rate
        self.crate = crossover_rate
        self.max_mutations = max_mutations
        self.logfile = log_file
        if not resumed:
            log_file.write("Spice simulation commands:\n"+'\n'.join(self.spice_commands)+'\n\n\n')

        self.plot_titles = plot_titles
        self.plot_yrange = plot_yrange

        #Directory to save files in
        self.directory = title

        #Generate seed circuits
        seed = kwargs['seed']
        self.seed_circuits = []
        if seed!=None:
            if type(seed)==str:
                seed = [seed]
            for element in seed:
                self.seed_circuits.append(parse_circuit(element,self.parts_list))



    def parse_sim_options(self,option):
        """Parses spice simulation commands for ac,dc,trans and temp words.
        If ac simulation is found plotting scale is made logarithmic"""
        m = re.search(r'\n(ac|AC) [a-zA-Z1-9]* [0-9\.]* [0-9\.]* [0-9\.]*[a-zA-Z]?[0-9\.]*',option)
        temp = ('.temp' in option) or ('.dtemp' in option)
        if m!=None:
            m = m.group(0).split()
            return m[0],False if m[1]=='lin' else True,multipliers(m[-1]),temp
        m = re.search(r'\n(dc|DC) [a-zA-Z1-9]* [0-9\.]* [0-9\.]*',option)
        if m!=None:
            m = m.group(0).split()
            return m[0],False,multipliers(m[-1]),temp
        m = re.search(r'\n(tran|TRAN) [0-9\.]*[a-zA-Z]? [0-9\.]*[a-zA-Z]?',option)
        if m!=None:
            m = m.group(0).split()
            return m[0],False,multipliers(m[-1]),temp
        else:
            return 0,False,0,temp

    def rank_pool(self,pool):
        """Multithreaded version of self.rank, computes scores for whole pool"""
        scores  = [0 for i in xrange(self.pool_size)]

        lasterror = None
        timeouts = 0
        for i in xrange(len(self.spice_commands)):
            errors = 0
            skipped = 0
            for t in xrange(self.pool_size/THREADS):

                threads = []
                for e,a in enumerate(pool[t*THREADS:t*THREADS+THREADS]):
                    if scores[THREADS*t+e]<inf:
                        threads.append(circuits.spice_thread(a.spice_input(self.spice_commands[i])))
                    else:
                        threads.append(None)

                for e,thread in enumerate(threads):
                    if scores[THREADS*t+e]<inf:
                        thread.start()
                for e,thread in enumerate(threads):
                    if scores[THREADS*t+e]<inf:
                        thread.join(self.timeout)
                        if thread.is_alive():
                            timeouts += 1
                            try:
                                thread.spice.terminate()
                            except OSError:#Thread died before we could kill it
                                pass
                            thread.join()

                for e,thread in enumerate(threads):
                    if thread==None:
                        #Something went wrong, just ignore it
                        scores[THREADS*t+e]=inf
                        skipped+=1
                        continue
                    if scores[THREADS*t+e]>=inf:
                        #If score is more than infinity don't bother doing more runs
                        skipped+=1
                    else:
                        if thread.result==None:
                            #Error in simulation
                            errors+=1
                            lasterror = "Simulation timedout"
                            scores[THREADS*t+e]=inf
                        elif thread.result[1]=={}:
                            #SPICE didn't return anything
                            errors+=1
                            lasterror = thread.result[0]
                            scores[THREADS*t+e]=inf
                        else:
                            #Everything seems to be fine, calculate scores
                            if scores[THREADS*t+e]<inf:
                                #Custom scoring function
                                if self.c_rank!=None:
                                    scores[THREADS*t+e]+=self.c_rank(thread.result[1],extra=pool[THREADS*t+e].extra_value,circuit=pool[t*THREADS+e])
                                #Default scoring function
                                if self.def_scoring:
                                    for k in thread.result[1].keys():
                                        scores[THREADS*t+e]+=self._rank(thread.result[1],i,k,extra=pool[THREADS*t+e].extra_value,circuit=pool[t*THREADS+e])
                            thread.result = None
            #print 'Skipped: {}'.format(skipped)
            if errors + skipped == self.pool_size:
                #All simulations failed
                raise SyntaxError("Simulation {} failed for every circuit.\nSpice returned {}".format(i,lasterror))
        if timeouts != 0:
            print '{} simulation(s) timed out'.format(timeouts)
        if timeouts > len(self.pool)/10:
            if timeouts > len(self.pool)/2:
                self.timeout *= 1.5
                print "Increasing timeout length by 50%, to {}".format(self.timeout)
            else:
                self.timeout *= 1.25
                print "Increasing timeout length by 25%, to {}".format(self.timeout)
        return sorted([(scores[i],pool[i]) for i in xrange(self.pool_size)])

    def _rank(self,x,i,k,extra=None,circuit=None):
        """Score of single circuit against single fitness function
        x is a dictionary of measurements, i is number of simulation, k is the measurement to score"""
        total=0.0
        func = self.ff[i]
        try:#fitness_weight might be None, or it might be list of None, or list of dictionary that contains None
            weight = self.fitness_weight[i][k]
        except (KeyError,TypeError,IndexError):
            weight = lambda x,**kwargs:1
        #If no weight function create function that returns one for all inputs
        if type(weight) in (int,long,float):
            c = weight
            weight = lambda x,**kwargs:float(c)#Make a constant anonymous function

        try:
            f = x[k][0]#Input
            v = x[k][1]#Output
            y = float(max(f))
        except:
            return inf
        #Sometimes spice doesn't simulate whole frequency range
        #I don't know why, so I just check if spice returned the whole range
        if y<0.99*self.frange[i]:
            return inf


        con_filled = True
        con_penalty=0
        for p in xrange(1,len(f)):
            try:
                total+=weight( f[p],extra=extra, generation=self.generation)*(f[p]-f[p-1])*( func(f[p],k,extra=extra, generation=self.generation) - v[p] )**2
            except TypeError:
                print 'Fitness function returned invalid value'
                raise
            except OverflowError:
                self.overflowed += 1
                total=inf
                pass
            if self.constraints[i]!=None and self.c_free_gens<=self.generation :
                con=self.constraints[i]( f[p],v[p],k,extra=extra,generation=self.generation )
                if con==None:
                    print 'Constraint function {} return None, for input: ({},{},{},extra={},generation={})'.format(i,f[p],v[p],k,extra,self.generation)
                if con==False:
                    con_penalty+=1
                    con_filled=False

        total/=y
        if total<0:
            return inf
        if self.generation%20>17:
            if circuit!=None and con_filled:
                total+=sum([element.cost if hasattr(element,'cost') else 0.1 for element in circuit.elements])
        #if con_penalty>1e5:
        #    con_penalty=1e5
        if self.c_free_gens>self.generation:
            return 1000*total+self.constraint_weight[i]*10
        if self.gradual_constraints:
            if self.generation>=self.constraint_ramp-self.c_free_gens:
                m = 1.0
            else:
                m = (self.generation-self.c_free_gens)/float(self.constraint_ramp)
            total+=m*con_penalty*self.constraint_weight[i]/float(len(f))
        else:
            total+=con_penalty*self.constraint_weight[i]/float(len(f))
        return total*1000+20000*(not con_filled)

    def printpool(self):
        """Prints all circuits and their scores in the pool"""
        for f,c in self.pool:
            print f,c
        print

    def save_plot(self,circuit,i,name='',**kwargs):
        v = circuit.evaluate(self.spice_commands[i],self.timeout)[1]
        #For every measurement in results
        for k in v.keys():
            score = self._rank(v,i,k,extra=circuit.extra_value)

            plt.figure()
            freq = v[k][0]
            gain = v[k][1]
            goal_val = [self.ff[i](f,k,extra=circuit.extra_value,generation=self.generation) for f in freq]
            #if self.plot_weight:
            #    weight_val = [self.fitness_weight[i](c,k) for c in freq]
            if self.constraints[i]!=None and self.plot_constraints:
                constraint_val = [(freq[c],gain[c])  for c in xrange(len(freq)) if not self.constraints[i](freq[c],gain[c],k,extra=circuit.extra_value,generation=self.generation)]

            if self.log_plot[i]==True:#Logarithmic plot
                plt.semilogx(freq,gain,'g',basex=10)
                plt.semilogx(freq,goal_val,'b',basex=10)
                #if self.plot_weight:
                #    plt.semilogx(freq,weight_val,'r--',basex=10)
                if self.plot_constraints:
                    plt.plot(*zip(*constraint_val), marker='.', color='r', ls='')
            else:
                plt.plot(freq,gain,'g')
                plt.plot(freq,goal_val,'b')
                #if self.plot_weight:
                #    plt.plot(freq,weight_val,'r--')
                if self.plot_constraints:
                    plt.plot(*zip(*constraint_val), marker='.', color='r', ls='')

            # update axis ranges
            ax = []
            ax[0:4] = plt.axis()
            # check if we were given a frequency range for the plot
            if k in self.plot_yrange.keys():
                plt.axis([min(freq),max(freq),self.plot_yrange[k][0],self.plot_yrange[k][1]])
            else:
                plt.axis([min(freq),max(freq),min(-0.5,-0.5+min(goal_val)),max(1.5,0.5+max(goal_val))])

            if self.sim_type[i]=='dc':
                plt.xlabel("Input (V)")
            if self.sim_type[i]=='ac':
                plt.xlabel("Input (Hz)")
            if self.sim_type[i]=='tran':
                plt.xlabel("Time (s)")

            try:
                plt.title(self.plot_titles[i][k])
            except:
                plt.title(k)

            plt.annotate('Generation '+str(self.generation),xy=(0.05,0.95),xycoords='figure fraction')
            if score!=None:
                plt.annotate('Score '+'{0:.2f}'.format(score),xy=(0.75,0.95),xycoords='figure fraction')
            plt.grid(True)
            # turn on the minor gridlines to give that awesome log-scaled look
            plt.grid(True,which='minor')
            if len(k)>=3 and k[1:3] == 'db':
                plt.ylabel("Output (dB)")
            elif k[0]=='v':
                plt.ylabel("Output (V)")
            elif k[0]=='i':
                plt.ylabel("Output (A)")

            plt.savefig(path_join(self.directory,strftime("%Y-%m-%d %H:%M:%S")+'-'+k+'-'+name+'.png'))

    def step(self):
        self.generation+=1

        def sf(pool,weight=1):
            #Select chromosomes from pool using weighted probablities
            r=random.random()**weight
            return random.choice(pool[:1+int(len(pool)*r)])[1]

        #Update best
        if self.generation==1:
            #Randomly generate some circuits
            newpool=[Chromosome(self.max_parts,self.parts_list,self.nodes,extra_value=self.extra_value) for i in xrange(self.pool_size)]
            #Generate seed circuits
            for i,circuit in enumerate(self.seed_circuits):
                newpool[i].elements=circuit
                #Set seed circuit extra values
                if self.extra_value!=None:
                    best_value = (inf,None)
                    for c,value in enumerate(self.extra_value):
                        #10 test values for one extra_value
                        for x in xrange(11):
                            ev = value[0]+(value[1]-value[0])*(x/10.0)
                            newpool[i].extra_value[c] = ev
                            score = 0
                            for command in xrange(len(self.spice_commands)):
                                v = newpool[i].evaluate(self.spice_commands[command],self.timeout)[1]
                                for k in v.keys():
                                    score += self._rank(v,command,k,extra=newpool[i].extra_value)
                            if 0 < score < best_value[0]:
                                #Better value found
                                best_value = (score,ev)
                        #Set the extra value to the best one found
                        print "Seed circuit {}, best found extra_value {}: {}".format(i,c,best_value[1])
                        newpool[i].extra_value[c] = best_value[1]
                #No extra values
                score = 0
                for command in xrange(len(self.spice_commands)):
                    v = newpool[i].evaluate(self.spice_commands[command],self.timeout)[1]
                    for k in v.keys():
                        score += self._rank(v,command,k,extra=newpool[i].extra_value)
                print "Seed circuit {}, score: {}".format(i,score)
                self.plotbest(newpool,0)

        else:
            if self.elitism!=0:
                #Pick self.elitism amount of best performing circuits to the next generation
                newpool=[self.pool[i][1] for i in xrange(self.elitism)]
            else:
                newpool=[]

            #FIXME this should be enabled or disabled in the simulation settings
            #if (not self.constraints_filled) and (self.alltimebest[0]<10000):
            #    print 'Constraint filling solution found'
            #    print 'Optimizing for number of elements'
            #    self.constraints_filled = True
            self.best=self.pool[0]

            #We have already chosen "self.elitism" of circuits in the new pool
            newsize=self.elitism
            while newsize<(1.0-self.rnd_circuits)*self.pool_size:
                newsize+=1
                c=deepcopy(sf(self.pool,weight=self.selection_weight))   #selected chromosome
                if random.random()<=self.crate:#crossover
                    d=sf(self.pool,weight=self.selection_weight)
                    l = max(len(c.elements),len(d.elements))
                    r1 = random.randint(0,l)
                    r2 = random.randint(0,l)
                    if r1>r2:
                        r1,r2=r2,r1
                    c.elements = c.elements[:r1]+d.elements[r1:r2]+c.elements[r2:]

                if random.random()<=self.mrate:#mutation
                    c.mutate()
                    tries=0
                    while random.random()<=self.mrate and tries<self.max_mutations:
                        tries+=1
                        c.mutate()
                newpool.append(c)
            while newsize<self.pool_size:
                #Generate new circuits randomly
                newpool.append(Chromosome(self.max_parts,self.parts_list,self.nodes,extra_value=self.extra_value))
                newsize+=1
        start = time()
        self.pool = self.rank_pool(newpool)
        #for i in self.pool:
        #    print i

        if (self.overflowed > self.pool_size/10):
            print "{0}\% of the circuits score overflowed, can't continue reliably. Try to decrease scoring weights.".format(100*float(self.overlflowed)/self.pool_size)
            exit()

        self.overflowed = 0

        print "Simulations per second: {}".format(round((len(self.spice_commands)*self.pool_size)/(time()-start),1))
        print "Time per generation: {} seconds".format(round(time()-start,1))
        if self.c_free_gens== self.generation:
            if self.constraints != None:
                print "Constraints enabled"
            self.alltimebest = (inf,None)

        if self.gradual_constraints and self.generation>self.c_free_gens:
            if self.generation-self.c_free_gens>self.constraint_ramp:
                self.gradual_constraints = False


        if self.plot_all_gens or self.pool[0][0]<self.alltimebest[0]:
                print strftime("%Y-%m-%d %H:%M:%S")
                if self.pool[0][1].extra_value != None:
                    print 'Extra values: '+str(self.pool[0][1].extra_value)
                print "Generation "+str(self.generation)+" New best -",self.pool[0][0],'\n',self.pool[0][1].pprint(),'\n'
                #print 'Cache size: %d/%d'%(self.cache_size,self.cache_max_size)+', Cache hits',self.cache_hits
                self.alltimebest=self.pool[0]
                self.plotbest()
                self.logfile.write(strftime("%Y-%m-%d %H:%M:%S")+' - Generation - '+str(self.generation) +' - '+str(self.alltimebest[0])+':\n'+self.alltimebest[1].pprint()+'\n\n')
                self.logfile.flush()#Flush changes to the logfile

        #Scale score for gradual constraint ramping
        if self.gradual_constraints and self.generation>self.c_free_gens:
            if self.generation-self.c_free_gens<self.constraint_ramp:
                self.alltimebest = (self.alltimebest[0]*(1-self.c_free_gens+self.generation)/float(self.generation-self.c_free_gens),
                        self.alltimebest[1])

    def averagefit(self):
        """Returns average score of the whole pool."""
        return sum(i[0] for i in self.pool)/float(self.pool_size)

    def plotbest(self,pool=None,nth=0):
        """Plot best circuit in the current pool.
        Alternatively plot nth circuit from the pool given in arguments."""
        if pool==None:
            pool = self.pool
        try:
                    circuit = pool[nth][1]
        except AttributeError:
                    #Not scored yet
                    circuit = pool[nth]
        try:
            if PLOTTING=='matplotlib':
                for c in xrange(len(self.spice_commands)):
                    self.save_plot(
                            circuit,
                            i=c,name=str(c))
            elif PLOTTING=='external':
                for i in xrange(len(self.spice_commands)):
                    v = circuit.evaluate(self.spice_commands[i],self.timeout)[1]
                    for k in v.keys():
                        freq = v[k][0]
                        gain = v[k][1]
                        score = self._rank(v,i,k,extra=circuit.extra_value)
                        goal_val = [self.ff[i](f,k,extra=circuit.extra_value,generation=self.generation) for f in freq]
                        if self.constraints[i]!=None and self.plot_constraints:
                            constraint_val = [(freq[c],gain[c])  for c in xrange(len(freq)) if not self.constraints[i](freq[c],gain[c],k,extra=circuit.extra_value,generation=self.generation)]
                        else:
                            constraint_val = None


                        plt = subprocess.Popen(['python','plotting.py'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                        try:
                            title = self.plot_titles[i][k]
                        except (KeyError,TypeError):
                            title = None

                        if k in self.plot_yrange.keys():
                            yrange = self.plot_yrange[k]
                        else:
                            yrange = None
                        data = ((freq,gain),k,goal_val,self.sim_type[i],self.generation,score,self.directory,title,yrange,self.log_plot[i],constraint_val,i)
                        plt.communicate(str(data))
        except:
            print "Plotting failed"
            raise


    def run(self):
        try:
            while True:
                self.step()
                print "Saving progress"
                self.save_progress(path_join(self.directory,'.dump'))
        except KeyboardInterrupt:
            pass

    def save_progress(self,out):
        """Saves CGP pool,generation and log filename to file"""
        #pickle format: (generation,pool,logfile)
        out_temp=out+'.tmp'

        with open(out_temp,'w') as dump:
            data = (self.generation,self.pool,self.logfile.name)
            pickle.dump(data,dump)
            print "Saving done"
        try:
            os.remove(out)
        except OSError:
            pass#First time saving and file doesn't exist yet
        os.rename(out_temp,out)

def load_settings(filename):
    no_defaults= ['title','max_parts','spice_commands','parts','fitness_function']
    default_settings = {'common':'',
                        'models':'',
                        'constraints':None,
                        'population':1000,
                        'nodes':None,
                        'elitism':1,
                        'mutation_rate':0.75,
                        'crossover_rate':0.1,
                        'fitness_weight':None,
                        'extra_value':None,
                        'log_file':None,
                        'plot_titles':None,
                        'plot_yrange':None,
                        'selection_weight':1,
                        'constraint_weight':None,
                        'max_mutations':5,
                        'constraint_free_generations':1,
                        'gradual_constraints':True,
                        'constraint_ramp':20,
                        'random_circuits':0.01,
                        'plot_every_generation':False,
                        'default_scoring':True,
                        'custom_scoring':None,
                        'seed':None,
                        'timeout':1.0,
                        }
    settings = default_settings.copy()
    temp = {}
    execfile(filename,temp)
    for key in temp:
        if key[0]!='_':
            if key not in default_settings.keys()+no_defaults:
                print 'Ignoring unknown command:',key
                continue
            settings[key] = temp[key]
    for key in no_defaults:
        if key not in settings.keys():
            print '"{}" is required'.format(key)
            exit()
    print 'Settings loaded'
    return settings

def main(filename):
    settings = load_settings(filename)
    if not os.path.exists(settings['title']):
        os.makedirs(settings['title'])
    resume = False
    try:
        r_file = open(path_join(settings['title'],'.dump'),'r')
        print "Do you want to resume Y/n:"
        while True:
            r = getch._Getch()()
            if r=='':
                resume = True
                print "Resumed"
                break
            if r in ('y','Y','\n'):
                resume = True
                print "Resumed"
                break
            if r in ('n','N'):
                break
    except IOError:
        print 'No save file found'
        pass

    if resume:
        #Resuming from file
        resumed = pickle.load(r_file)
        outfile = open(resumed[2],'a')
    if settings['log_file']!=None:
        settings['log_file']=outfile
    else:
        if resume:
            settings['log_file']= open(resumed[2],'a')
        else:
            settings['log_file']= open(path_join(settings['title'],'sim'+strftime("%Y-%m-%d %H:%M:%S")+'.log'),'w')



    e = CGP(resume=resume,**settings)

    if resume:
        e.generation = resumed[0]
        e.pool = resumed[1]
    e.run()

if __name__ == '__main__':
    main(sys.argv[1])
