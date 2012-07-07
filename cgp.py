import circuits
import random
from copy import deepcopy
from time import strftime
import math
import re
import pdb
import matplotlib.pyplot as plt
import hashlib
import pickle
import getch
inf = 1e12
simulation_timeout = 0.5#seconds

class Circuit_gene:
    """Represents single circuit element"""
    def __init__(self,name,nodes,*args):
        self.spice_name = name
        self.name = name+str(id(self))
        self.values = args
        self.spice_options = ' '.join(map(str,*args))
        self.nodes = nodes
        self.nodesstr = ' '.join(map(str,nodes))

    def __repr__(self):
        return self.name+' '+self.nodesstr+' '+self.spice_options

    def spice(self):
        return self.name+' '+self.nodesstr+' '+self.spice_options

def log_dist(a,b):
    #Uniform random number in logarithmic scale
    return random.uniform(1,10)*(10**(random.uniform(a,b)))

def same(x):
    #True if all elements are same
    return reduce(lambda x,y:x==y,x)

def random_element(parts,node_list):
    #Return random circuit element from parts list
    name = random.choice(parts.keys())
    part = parts[name]
    spice_line = []
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
    def __init__(self,max_parts,parts_list,nodes):
        self.max_parts = max_parts
        self.nodes = ['n'+str(i) for i in xrange(nodes)]
        self.nodes.append('0')
        self.parts_list = parts_list
        self.elements = [random_element(self.parts_list,self.nodes) for i in xrange(random.randint(1,max_parts/2))]

    def __repr__(self):
        return str(self.elements)

    def pprint(self):
        return '\n'.join(map(str,self.elements))

    def mutate(self):
        m = random.randint(0,4)
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

    def evaluate(self,options):
        global simulation_timeout
        program = options+'\n'
        for i in self.elements:
            program+=i.spice()+'\n'
        #FIXME check if n2 is in node list
        #Should need to check that there exists a path from input to output
        if ' n2 ' not in program:
            return None
        try:
            out = circuits.simulate(program,simulation_timeout)
        except circuits.Timeout:
            print 'timeout'
            return None
        return out

def multipliers(x):
    """Convert values with units to values"""
    try:
        return float(x)
    except:
        pass
    a = x[-1]
    y = float(x[:-1])
    endings = {'u':-6,'n':-9,'p':-12,'s':0}
    return y*(10**endings[a])

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
    plot_titles=None,
    plot_yrange=None):
        self.spice_commands=spice_sim_commands
        sim = map(self.parse_sim_options,self.spice_commands)

        print sim
        for e,i in enumerate(sim,1):
            print 'Simulation {0} - Type: {1}, Logarithmic plot: {2}'.format(e,i[0],i[1])
            if i[3]:
                print 'Temperature specified in simulation'
                self.temperatures = True
                #TODO write current temperature in the plot

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

        if all(i==None for i in self.fitness_weight):
            self.plot_weight=False
        else:
            self.plot_weight=True

        if all(i==None for i in self.constraints):
            self.plot_constraints=False
        else:
            self.plot_constraints=True

        if len(self.spice_commands)>len(self.fitness_weight):
            raise Exception('Fitness function weight list length is incorrect')
        if len(self.spice_commands)>len(self.ff):
            raise Exception('Not enough fitness functions')

        self.pool_size=pool_size
        self.parts_list = parts_list
        self.generation=1
        self.elitism=elitism
        self.alltimebest=(float('inf'),float('inf'))
        self.mrate = mutation_rate
        self.crate = crossover_rate
        self.logfile = log
        log.write("Spice simulation command:\n"+'\n'.join(self.spice_commands)+'\n\n\n')

        #Create pool of random circuits
        temp=[Chromosome(max_parts,parts_list,nodes) for i in xrange(pool_size)]
        self.pool=sorted([ (self.rank(c),c) for c in temp])
        self.best=(self.generation,self.pool[0])
        self.history=[(self.generation,self.best[0],self.averagefit())]

        self.plot_titles = plot_titles
        self.plot_yrange = plot_yrange


    def parse_sim_options(self,option):
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

    def rank(self,c):
        """Calculate score of single circuit"""
        #c = circuit
        h = hashlib.md5(str(c)).hexdigest()
        if h in self.cache:
            self.cache_hits +=1
            return self.cache[h]
        total=0.0
        for i in xrange(len(self.spice_commands)):
            x = c.evaluate(self.spice_commands[i])
            if x==None or len(x.keys())==0:
                return inf

            for k in x.keys():
                total+=self._rank(x,i,k)
        if self.cache_size<self.cache_max_size:
            self.cache_size+=1
            self.cache[h]=total
        #TODO fix constraints
        #if self.constraints_filled:
        #    total+=5*len(c.elements)
        #else:
        #    total+=1000
        return total

    def _rank(self,x,i,k,c=None):
        """Score of single circuit against single fitness function
        x is a dictionary of measurements, i is number of simulation, k is the measurement to score"""
        if c!=None:
            #Circuit in input and x needs to be calculated.
            x = c.evaluate(self.spice_commands[i])
            if x==None or len(x.keys())==0:
                return inf
        total=0.0
        func = self.ff[i]
        weight = self.fitness_weight[i]
        #If no weight function create function that returns one for all inputs
        if weight==None:
            weight = lambda x:1
        constraint = self.constraints[i]
        if constraint == None:
            constraint = lambda f,x,k : 0

        f = x[k][0]#Input
        v = x[k][1]#Output
        y = float(max(f))
        #Sometimes spice doesn't simulate whole frequency range
        #I don't know why, so I just check if spice returned the whole range
        if y<0.99*self.frange[i]:
            return inf
        con_filled = True
        if self.log_plot[i]:
            for p in xrange(1,len(f)):
                #Divided by frequency for even scores across whole frequency range in log scale.
                try:
                    total+=weight( (f[p]-f[p-1]/2) )*(x[0][p]-x[0][p-1])*(( func(f[p],k)+func(f[p-1],k) - x[1][p] - x[1][p-1])**2)/x[0][p]
                except TypeError:
                    print 'Fitness function returned invalid value'
                    raise

                if not constraint( f[p],v[p],k ):
                    con_filled=False
        else:
            for p in xrange(1,len(f)):
                try:
                    total+=weight( (f[p]-f[p-1])/2 )*(f[p]-f[p-1])*( func(f[p],k) + func(f[p-1],k) - v[p] - v[p-1] )**2
                except TypeError:
                    print 'Fitness function returned invalid value'
                    raise
                if not constraint( f[p],v[p],k ):
                    con_filled=False

            total/=y
        if total<0:
            return inf
        #FIXME constraints don't really work anymore after adding multiple measurement per simulation
        #Constraints are still assigned per simulation, not per measurement
        return total*1000#+10000*(not con_filled)

    def printpool(self):
        for f,c in self.pool:
            print f,c
        print

    def save_plot(self,circuit,i,log=True,name='',**kwargs):
        v = circuit.evaluate(self.spice_commands[i])
        #For every measurement in results
        for k in v.keys():
            score = self._rank(v,i,k)

            plt.figure()
            freq = v[k][0]
            gain = v[k][1]
            goal_val = [self.ff[i](c,k) for c in freq]
            if self.plot_weight:
                weight_val = [self.fitness_weight[i](c,k) for c in freq]
            if self.constraints[i]!=None and self.plot_constraints:
                constraint_val = [not self.constraints[i](freq[c],gain[c],k) for c in xrange(len(freq))]
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
                if self.constraints[i]!=None and self.plot_constraints:
                    plt.plot(freq,constraint_val,'m')

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

            if self.plot_titles!=None:
                #FIXME Plot tiles are too complex when circuits return more than one measurement
                #plt.title(self.plot_titles[i])
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

            plt.savefig('plot'+strftime("%Y-%m-%d %H:%M:%S")+'-'+k+'-'+name+'.png')

    def step(self):
        self.generation+=1

        def sf(pool):
            #Select chromosomes from pool using weighted propablities
            r=random.random()
            return random.choice(pool[:1+int(len(pool)*r)])[1]

        #Update best
        if self.elitism!=0:
            if self.pool[0][0]<self.alltimebest[0]:
                print "Generation "+str(self.generation)+" New best -",self.pool[0][0],'\n',self.pool[0][1].pprint(),'\n'
                print 'Cache size: %d/%d'%(self.cache_size,self.cache_max_size)+', Cache hits',self.cache_hits
                self.alltimebest=self.pool[0]
                self.plotbest()
                self.logfile.write(strftime("%Y-%m-%d %H:%M:%S")+' - Generation - '+str(self.generation) +' - '+str(self.alltimebest[0])+':\n'+self.alltimebest[1].pprint()+'\n\n')
            newpool=[self.pool[i][1] for i in xrange(self.elitism)]
        else:
            #elif self.generation%5==0:
            print "Generation "+str(self.generation)+" Current best -",self.pool[0][0],'\n',self.pool[0][1].pprint(),'\n'
            print 'Cache size: %d/%d'%(self.cache_size,self.cache_max_size)+', Cache hits',self.cache_hits
            self.plotbest()
            self.logfile.write(strftime("%Y-%m-%d %H:%M:%S")+' - '+str(self.pool[0][0])+':\n'+self.pool[0][1].pprint()+'\n\n')
            newpool=[]

        self.logfile.flush()#Flush changes to the logfile
        #FIXME constraints still don't work
        #if (not self.constraints_filled) and (self.alltimebest[0]<10000):
        #    print 'Constraint filling solution found'
        #    print 'Optimizing for number of elements'
        #    self.constraints_filled = True
        self.best=self.pool[0]
        self.history.append((self.generation,self.best[0],self.averagefit()))

        #Choose self.elitism best chromosomes to next generation
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
        self.pool=sorted([(self.rank(i),i) for i in newpool])

    def averagefit(self):
        return sum(i[0] for i in self.pool)/float(self.pool_size)

    def plotbest(self):
        try:
            for c in xrange(len(self.spice_commands)):
                self.save_plot(
                        self.pool[0][1],
                        i=c,log=self.log_plot,name=str(c))
        except ValueError:
            print 'Plotting failed'

def gaussian(x,mean,std):
    return math.exp(-(x-mean)**2/(2*std**2))

def weight(x,center=100):
    return 0.5+gaussian(x,100,30)

def goal(freq):
    return 5 if freq<2.5 else 0

options=[
"""
.control
dc Vin 4 10 0.1
print v(n2)
.endc
$.MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
$.MODEL QMOD2 PNP(BF=100 CJC=20pf CJE=20pf IS=1E-16)
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
*
*               Fairchild  pid=66   case=TO92
*               11/19/2001 calccb update
*$
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
*               Fairchild        pid=23          case=TO92
*               88-09-08 bam    creation
.SUBCKT 1N4148 1 2
*
* The resistor R1 does not reflect
* a physical device. Instead it
* improves modeling in the reverse
* mode of operation.
*
R1 1 2 5.827E+9
D1 1 2 1N4148
*
.MODEL 1N4148 D
+ IS = 4.352E-9
+ N = 1.906
+ BV = 110
+ IBV = 0.0001
+ RS = 0.6458
+ CJO = 7.048E-13
+ VJ = 0.869
+ M = 0.03
+ FC = 0.5
+ TT = 3.48E-9
.ENDS
Vin n1 0
rload 100k n2 0
""",
"""
.control
dc Vin 4 10 0.1
print v(n2)
print i(Vin)
.endc
.temp 60
$.MODEL QMOD NPN(BF=100 CJC=20pf CJE=20pf IS=1E-16)
$.MODEL QMOD2 PNP(BF=100 CJC=20pf CJE=20pf IS=1E-16)
.model 2N3906  PNP(Is=455.9E-18 Xti=3 Eg=1.11 Vaf=33.6 Bf=204 Ise=7.558f
+               Ne=1.536 Ikf=.3287 Nk=.9957 Xtb=1.5 Var=100 Br=3.72
+               Isc=529.3E-18 Nc=15.51 Ikr=11.1 Rc=.8508 Cjc=10.13p Mjc=.6993
+               Vjc=1.006 Fc=.5 Cje=10.39p Mje=.6931 Vje=.9937 Tr=10n Tf=181.2p
+               Itf=4.881m Xtf=.7939 Vtf=10 Rb=10, level=1)
*
*               Fairchild  pid=66   case=TO92
*               11/19/2001 calccb update
*$
.model 2N3904   NPN(Is=6.734f Xti=3 Eg=1.11 Vaf=74.03 Bf=416.7 Ne=1.259
+               Ise=6.734f Ikf=66.78m Xtb=1.5 Br=.7371 Nc=2 Isc=0 Ikr=0 Rc=1
+               Cjc=3.638p Mjc=.3085 Vjc=.75 Fc=.5 Cje=4.493p Mje=.2593 Vje=.75
+               Tr=239.5n Tf=301.2p Itf=.4 Vtf=4 Xtf=2 Rb=10, level=1)
*               Fairchild        pid=23          case=TO92
*               88-09-08 bam    creation
.SUBCKT 1N4148 1 2
*
* The resistor R1 does not reflect
* a physical device. Instead it
* improves modeling in the reverse
* mode of operation.
*
R1 1 2 5.827E+9
D1 1 2 1N4148
*
.MODEL 1N4148 D
+ IS = 4.352E-9
+ N = 1.906
+ BV = 110
+ IBV = 0.0001
+ RS = 0.6458
+ CJO = 7.048E-13
+ VJ = 0.869
+ M = 0.03
+ FC = 0.5
+ TT = 3.48E-9
.ENDS
Vin n1 0
rload 100k n2 0
"""]


#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':1,'min':0,'max':7},
         'C':{'nodes':2,'value':1,'min':-13,'max':-3},
         #'L':{'nodes':2,'value':1,'min':-9,'max':-3},
         #'D1':{'nodes':2,'spice':'1N4148'},
         'Q1':{'nodes':3,'spice':'2N3906'},
         'Q2':{'nodes':3,'spice':'2N3904'}
         }


def weight1(f):
    n = 10**(-9)
    if f<=81*n or f>(140*n):
        return 1
    if 120*n<=f<140*n:
        return 1
    if f>=190*n:
        return 2
    return 0.2

def weight2(f):
    n = 10**(-9)
    if f<=31*n or f>(130*n):
        return 1
    if 110*n<=f<130*n:
        return 1
    if f>=180*n:
        return 2
    return 0.2

def constraint1(f,x,k):
    """f is Input, x is Output, k is the name of measurement"""
    if k[0]=='i':
        return x<0.01

def constraint2(f,x):
    n= 10**(-9)
    if 0<=f<19*n:
        if x>1 or x<-1:
            return False
    if 25*n<f<=28*n:
        if x<3.5:
            return False
    if 110*n<f<=119*n or f>=140*n:
        if x>1 or x<-1:
            return False
    if 125*n<=f<128*n:
        if x>0.5:
            return False
    return True

def goal1(f,k):
    """k is the name of measurement. eq. v(n2)"""
    if k=='v(n2)':
        return 2.5
    elif k[0]=='i':
        return 0

def goal2(f,k):
    n= 10**(-9)
    if 22*n<=f<=32*n:
        return 5
    #if 122*n<=f<=132*n:
    #    return 5
    return 0


if __name__ == "__main__":
    resume = False
    try:
        r_file = open('.dump','r')
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
        pass

    if not resume:
        outfile = open('sim'+strftime("%Y-%m-%d %H:%M:%S")+'.log','w')
        e = CGP(pool_size=4000,
                nodes=12,
                parts_list=parts,
                max_parts=15,
                elitism=1,
                mutation_rate=0.7,
                crossover_rate=0.25,
                fitnessfunction=[goal1,goal1],
                fitness_weight=[None,None],
                constraints=[None,None],
                spice_sim_commands=options,
                log=outfile,
                plot_titles=["Output voltage(27C)","Output voltage(60C)"],
                plot_yrange={'v(n2)':(1,4),'i(vin)':(-3.0,0.5)})
    else:
        #Resuming from file
        e = pickle.load(r_file)
        e.logfile = open(e.logfile,'a')
        outfile = e.logfile
    try:
        while True:
            e.step()
    except KeyboardInterrupt:
        #Save space by erasing cache
        print "Saving state..."
        try:
            e.cache = {}
            e.cache_size = 0
            e.cache_hits = 0
            #Files can't be pickled. So save filename and open it again when unpickling.
            e.logfile = outfile.name
            dump = open('.dump','w')
            pickle.dump(e,dump)
        except KeyboardInterrupt:
            print "Saving state, please wait."
