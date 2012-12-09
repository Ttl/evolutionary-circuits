from math import log

title = 'Low-pass'

#Put SPICE device models used in the simulations here.

#This is for constructions common for every simulation

#List of simulations
#Put SPICE simulation options between.control and .endc lines
#
spice_commands=[
"""
.control
ac dec 30 10 1e5
print vdb(n2)
option rshunt = 1e12
.endc
Vin n1 0 ac 1
Rload n2 0 {aunif(1k,500)}
"""
]

#Dictionary of the availabe parts
#nodes: number of terminals
#value: Device has a value, eg. resistance, inductance
#min and max: Range for value.
#spice: This is added after anything else, can be used to specify device model
#or some other options for the device
parts = {'R':{'nodes':2,'value':1,'min':1,'max':1e6},
         'C':{'nodes':2,'value':1,'min':1e-10,'max':1e-3},
         'L':{'nodes':2,'value':1,'min':1e-9,'max':1e-3},
         }

def _fitness_function1(f,k,**kwargs):
    """k is the name of measurement. eq. v(n2)"""
    if k[0]=='v':
        #-100dB/decade
        return -43.43*log(f)+300 if f>=1000 else 0
    elif k[0]=='i':
        #Goal for current use is 0
        return 0

def _constraint1(f,x,k,**kwargs):
    if k[0]=='v':
        if f>1000:
            return x<=0
        elif f<100:
            return -3<x<3
        else:
            return -0.5<x<0.5
    return True

#This circuit will be added to the first generation
#Circuit below scores poorly, because it fails to fulfill the constraints
seed = """
R1 n1 n2 1k
C1 n2 0 150n
"""

population=2000#Too small population might not converge, or converges to local minimum, but is faster to simulate
max_parts=8#Maximum number of parts
#Enabling this makes the program ignore constraints for few first generations.
#Which makes the program try to fit right side first ignoring pass-band.
gradual_constraints = False
mutation_rate=0.75
crossover_rate=0.05
#selection_weight=1.5
fitness_function=[_fitness_function1,_fitness_function1]
fitness_weight=[{'vdb(n2)':lambda x,**kwargs:100 if x<3e3 else 0.1}]
constraints=[_constraint1]
constraint_weight=[10000]
plot_yrange={'vdb(n2)':(-120,20),'i(vin)':(-0.2,0.2),'i(vc)':(-0.2,0.2),'i(ve)':(-0.2,0.2),'v(n2)':(-2,2)}
