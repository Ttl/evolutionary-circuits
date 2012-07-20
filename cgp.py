import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import circuits
import random
from copy import deepcopy
from time import strftime,time
import re
import pickle
import getch
from os.path import join as path_join
import os
inf = 1e12
simulation_timeout = 0.5#seconds
THREADS = 4

class Circuit_gene:
    """Represents a single component"""
    def __init__(self,name,nodes,*args):
        #Name of the component(eg. "R1")
        self.spice_name = name
        #N-tuple of values
        self.values = args
        self.spice_options = ' '.join(map(str,*args))
        self.nodes = nodes
    def __repr__(self):
        return self.spice_name+str(id(self))+' '+' '.join(map(str,self.nodes))+' '+self.spice_options


def log_dist(a,b):
    #Uniform random number in logarithmic scale
    return random.uniform(1,10)*(10**(random.uniform(a,b)))

def same(x):
    #True if all elements are same
    return reduce(lambda x,y:x==y,x)

def random_element(parts,nodes):
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
    if 'spice' in part:
        spice_line.append(part['spice'])
    return Circuit_gene(name,nodes,spice_line)

def mutate_value(element,parts):
    val = element.values[0]
    name = element.spice_name
    try:
        val[0] = log_dist(parts[name]['min'],parts[name]['max'])
    except:
        return element
    return Circuit_gene(element.spice_name,element.nodes,val)

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

    def mutate(self):
        m = random.randint(0,5)
        i = random.randint(0,len(self.elements)-1)
        if m==0:
            #Change value of one component
            self.elements[i] = mutate_value(self.elements[i],self.parts_list)
        elif m==1:
            #Add one component if not already maximum number of components
            if len(self.elements)<self.max_parts:
                self.elements.append(random_element(self.parts_list,self.nodes))
        elif m==2 and len(self.elements)>1:
            #Delete one component
            del self.elements[i]
        elif m==3:
            #Delete one and add one component
            del self.elements[i]
            self.elements.append(random_element(self.parts_list,self.nodes))
        elif m==4:
            #Shuffle list of elements(better crossovers)
            random.shuffle(self.elements)
        elif m==5:
            #Change the extra_value
            if self.extra_range!=None:
                i = random.randint(0,len(self.extra_value)-1)
                self.extra_value[i] = random.uniform(*self.extra_range[i])

    def spice_input(self,options):
        """Generate the input to SPICE"""
        global simulatiion_timeout
        program = options+'\n'
        for i in self.elements:
            program+=str(i)+'\n'
        #FIXME, this shouldn't be hard coded
        #We should read from the spice commands all the
        #printed nodes
        if ' n2 ' not in program:
            return None
        return program

    def evaluate(self,options):
        """Used in plotting, when only 1 circuits needs to be simulated"""
        program = self.spice_input(options)
        if ' n2 ' not in program:
            return None
        thread = circuits.spice_thread(self.spice_input(options))
        thread.start()
        thread.join(simulation_timeout)
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
        endings = {'u':-6,'n':-9,'p':-12,'s':0}
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
    pool_size,
    nodes,
    parts_list,
    max_parts,
    elitism,
    mutation_rate,
    crossover_rate,
    fitnessfunction,
    fitness_weight,
    constraints,
    spice_sim_commands,
    log,
    directory='',
    resumed=False,
    extra_value=None,
    plot_titles=None,
    plot_yrange=None):
        self.spice_commands=spice_sim_commands
        sim = map(self.parse_sim_options,self.spice_commands)

        print strftime("%Y-%m-%d %H:%M:%S")
        for e,i in enumerate(sim,1):
            print 'Simulation {0} - Type: {1}, Logarithmic plot: {2}'.format(e,i[0],i[1])
            if i[3]:
                print 'Temperature specified in simulation'
                self.temperatures = True
                #TODO write the current temperature in the plot
        print

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
        self.fitness_weight = fitness_weight
        self.ff=fitnessfunction
        self.constraints = constraints
        self.constraints_filled = False

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

        if len(self.spice_commands)>len(self.fitness_weight):
            raise Exception('Fitness function weight list length is incorrect')
        if len(self.spice_commands)>len(self.ff):
            raise Exception('Not enough fitness functions')

        self.pool_size=pool_size-pool_size%THREADS+THREADS
        self.parts_list = parts_list
        self.generation=1
        self.elitism=elitism
        self.alltimebest=(float('inf'),float('inf'))
        self.mrate = mutation_rate
        self.crate = crossover_rate
        self.logfile = log
        if not resumed:
            log.write("Spice simulation command:\n"+'\n'.join(self.spice_commands)+'\n\n\n')
            temp=[Chromosome(max_parts,parts_list,nodes,extra_value=extra_value) for i in xrange(self.pool_size)]
            self.pool = self.rank_pool(temp)
            #self.pool=sorted([ (self.rank(c),c) for c in temp])
            del temp
            self.best=(self.generation,self.pool[0])

        self.plot_titles = plot_titles
        self.plot_yrange = plot_yrange

        #Directory to save files in
        self.directory = directory



    def parse_sim_options(self,option):
        """Parses spice simulation commands for ac,dc,trans and temp words.
        If ac simulation is found plotting scale is made logarithmic"""
        m = re.search(r'\n(ac|AC) [a-zA-Z1-9]* [0-9\.]* [0-9\.]* [0-9\.]*[a-zA-Z]?',option)
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
        results = [[None for c in xrange(len(self.spice_commands))] for i in xrange(self.pool_size)]
        threads = [None for i in xrange(THREADS)]

        lasterror = None
        for i in xrange(len(self.spice_commands)):
            errors = 0
            for t in xrange(self.pool_size/THREADS):

                threads = [circuits.spice_thread(a.spice_input(self.spice_commands[i])) for a in pool[t*THREADS:t*THREADS+THREADS]]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(simulation_timeout)
                    if thread.is_alive():
                        try:
                            thread.spice.terminate()
                        except OSError:#Thread died before we could kill it
                            pass
                        thread.join()

                for e,thread in enumerate(threads):
                    if thread.result==None:
                        errors+=1
                        lasterror = "Simulation timedout"
                    elif thread.result[1]=={}:
                        errors+=1
                        lasterror = thread.result[0]
                    else:
                        results[THREADS*t+e][i] = thread.result[1]
                        thread.result = None
                del threads
            if errors == self.pool_size:
                #All simulations failed
                raise SyntaxError("Simulation {} failed for every circuit.\nSpice returned {}".format(i,lasterror))

        #new_pool = [None for i in xrange(self.pool_size)]
        for t in xrange(self.pool_size):
            #new_pool[t] = [0,pool[t]]#(Score, Circuit)
            pool[t]=[0,pool[t]]
            for i in xrange(len(self.spice_commands)):
                if results[t][i]==None or len(results[t][i].keys())==0:
                    pool[t][0]=inf
                    continue
                for k in results[t][i].keys():
                    pool[t][0]+=self._rank(results[t][i],i,k,extra=pool[t][1].extra_value)
            pool[t]=tuple(pool[t])#Make immutable for reduction in memory

        return sorted(pool)

    def _rank(self,x,i,k,extra=None,c=None):
        """Score of single circuit against single fitness function
        x is a dictionary of measurements, i is number of simulation, k is the measurement to score"""
        if c!=None:
            #Circuit in input and x needs to be calculated.
            x = c.evaluate(self.spice_commands[i])
            if x==None or len(x.keys())==0:
                return inf
        total=0.0
        func = self.ff[i]
        try:#fitness_weight migh be None, or it might be list of None, or list of dictionary that contains None
            weight = self.fitness_weight[i][k]
        except IndexError:
            weight = lambda x:1
        except KeyError:
            weight = lambda x:1
        except TypeError:
            weight = lambda x:1
        #If no weight function create function that returns one for all inputs
        if type(weight) in (int,long,float):
            c = weight
            weight = lambda x:float(c)#Make a constant anonymous function

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
        if self.log_plot[i]:
            for p in xrange(1,len(f)):
                #Divided by frequency for even scores across whole frequency range in log scale.
                try:
                    total+=weight( (f[p]-f[p-1]/2) )*(x[0][p]-x[0][p-1])*(( func(f[p],k,extra=extra)+func(f[p-1],k,extra=extra) - x[1][p] - x[1][p-1])**2)/x[0][p]
                except TypeError:
                    print 'Fitness function returned invalid value'
                    raise

                if not constraint( f[p],v[p],k ):
                    con_filled=False
        else:
            for p in xrange(1,len(f)):
                try:
                    #total+=weight( (f[p]-f[p-1])/2 )*(f[p]-f[p-1])*( func(f[p],k) + func(f[p-1],k) - v[p] - v[p-1] )**2
                    total+=weight( f[p] )*(f[p]-f[p-1])*( func(f[p],k,extra=extra) - v[p] )**2
                except TypeError:
                    print 'Fitness function returned invalid value'
                    raise
                except OverflowError:
                    total=inf
                    pass
                if self.constraints[i]!=None:
                    con=self.constraints[i]( f[p],v[p],k,extra=extra,generation=self.generation )
                    if con==None:
                        print 'Constraint function {} return None'.format(i)
                    if con==False:
                        con_penalty+=100
                        con_filled=False

            total/=y
        if total<0:
            return inf
        if con_penalty>1e5:
            con_penalty=1e5
        total+=con_penalty
        return total*1000+10000*(not con_filled)

    def printpool(self):
        """Prints all circuits and their scores in the pool"""
        for f,c in self.pool:
            print f,c
        print

    def save_plot(self,circuit,i,log=True,name='',**kwargs):
        v = circuit.evaluate(self.spice_commands[i])[1]
        #For every measurement in results
        for k in v.keys():
            score = self._rank(v,i,k,extra=circuit.extra_value)

            plt.figure()
            freq = v[k][0]
            gain = v[k][1]
            goal_val = [self.ff[i](f,k,extra=circuit.extra_value,generation=self.generation) for f in freq]
            if self.plot_weight:
                weight_val = [self.fitness_weight[i](c,k) for c in freq]
            if self.constraints[i]!=None and self.plot_constraints:
                constraint_val = [not self.constraints[i](freq[c],gain[c],k,extra=circuit.extra_value,generation=self.generation) for c in xrange(len(freq))]
            if log==True:#Logarithmic plot
                plt.semilogx(freq,gain,'g',basex=10)
                plt.semilogx(freq,goal_val,'b',basex=10)
                if self.plot_weight:
                    plt.semilogx(freq,weight_val,'r--',basex=10)
                if self.plot_constraints:
                    plt.semilogx(freq,constraint_val,'m',basex=10)
            else:
                plt.plot(freq,gain,'g')
                plt.plot(freq,goal_val,'b')
                if self.plot_weight:
                    plt.plot(freq,weight_val,'r--')
                #FIXME: Need a better way to plot constraints
                #if self.constraints[i]!=None and self.plot_constraints:
                #    plt.plot(freq,constraint_val,'m')

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
            if k[0]=='v':
                plt.ylabel("Output (V)")
            elif k[0]=='i':
                plt.ylabel("Output (A)")

            plt.savefig(path_join(self.directory,strftime("%Y-%m-%d %H:%M:%S")+'-'+k+'-'+name+'.png'))

    def step(self):
        self.generation+=1

        def sf(pool):
            #Select chromosomes from pool using weighted probablities
            r=random.random()
            return random.choice(pool[:1+int(len(pool)*r)])[1]

        #Update best
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
        while newsize<self.pool_size:
            newsize+=1
            c=deepcopy(sf(self.pool))   #selected chromosome
            if random.random()<=self.crate:#crossover
                d=sf(self.pool)
                l = max(len(c.elements),len(d.elements))
                r1 = random.randint(0,l)
                r2 = random.randint(0,l)
                if r1>r2:
                    r1,r2=r2,r1
                c.elements = c.elements[:r1]+d.elements[r1:r2]+c.elements[r2:]

            if random.random()<=self.mrate:#mutation
                c.mutate()
                tries=0
                while random.random()<=self.mrate and tries<10:
                    tries+=1
                    c.mutate()
            newpool.append(c)
        start = time()
        self.pool = self.rank_pool(newpool)

        print "Simulations per second: {}".format(round((len(self.spice_commands)*self.pool_size)/(time()-start),1))
        print "Time per generation: {} seconds".format(round(time()-start,1))
        if self.pool[0][0]<self.alltimebest[0]:
                print strftime("%Y-%m-%d %H:%M:%S")
                print 'Extra values: '+str(self.pool[0][1].extra_value)
                print "Generation "+str(self.generation)+" New best -",self.pool[0][0],'\n',self.pool[0][1].pprint(),'\n'
                #print 'Cache size: %d/%d'%(self.cache_size,self.cache_max_size)+', Cache hits',self.cache_hits
                self.alltimebest=self.pool[0]
                self.plotbest()
                self.logfile.write(strftime("%Y-%m-%d %H:%M:%S")+' - Generation - '+str(self.generation) +' - '+str(self.alltimebest[0])+':\n'+self.alltimebest[1].pprint()+'\n\n')
                self.logfile.flush()#Flush changes to the logfile


    def averagefit(self):
        """Returns average score of the whole pool."""
        return sum(i[0] for i in self.pool)/float(self.pool_size)

    def plotbest(self):
        try:
            for c in xrange(len(self.spice_commands)):
                self.save_plot(
                        self.pool[0][1],
                        i=c,log=self.log_plot,name=str(c))
        except:
            print 'Plotting failed'
            raise

    def run(self):
        try:
            while True:
                self.step()
                print "Saving progress"
                self.save_progress(path_join(self.directory,'.dump'))
        except KeyboardInterrupt:
            print "Saving state..."
            self.save_progress(path_join(self.directory,'.dump'))

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
