from math import log

title = 'Low-pass filter'

#List of simulations
#Put SPICE simulation options between.control and .endc lines
#
spice_commands=[
"""
.control
ac dec 30 10 1e5
print vdb(out)
option rshunt = 1e12
.endc
Vin in 0 ac 1
Rload out 0 {aunif(10k,500)}
cload out 0 100p
"""
]

inputs = ['in']
outputs = ['out']

#Dictionary of the availabe parts
parts = {'R':{'nodes':2,'value':(0.1,1e6),},
         'C':{'nodes':2,'value':(1e-12,1e-5)},
         'L':{'nodes':2,'value':(1e-9,1e-3)},
         }

def _fitness_function1(f,k,**kwargs):
    """k is the name of measurement. eq. v(out)"""
    if k[0]=='v':
        #-100dB/decade
        return -43.43*log(f)+300 if f>=1000 else 0
    elif k[0]=='i':
        #Goal for current use is 0
        return 0

def _constraint1(f,x,k,**kwargs):
    if k[0]=='v':
        if f>8000:
            return x <= -20
        if f>1000:
            return x<=0.5
        elif f<100:
            return -5<x<3
        else:
            return -2<x<1
    return True

#This circuit will be added to the first generation
#Circuit below scores poorly, because it fails to fulfill the constraints
seed = """
R1 in out 1k
C1 out 0 150n
"""

population=3000#Too small population might not converge, or converges to local minimum, but is faster to simulate
max_parts=8#Maximum number of parts

#Enabling this makes the program ignore constraints for few first generations.
#Which makes the program try to fit right side first ignoring the pass-band.
gradual_constraints = False
mutation_rate=0.75
crossover_rate=0.10
#selection_weight=1.5
fitness_function=[_fitness_function1,_fitness_function1]
fitness_weight=[{'vdb(out)':lambda x,**kwargs:100 if x<3e3 else 0.1}]
constraints=[_constraint1]
constraint_weight=[10000]
plot_yrange={'vdb(out)':(-120,20),'i(vin)':(-0.2,0.2),'i(vc)':(-0.2,0.2),'i(ve)':(-0.2,0.2),'v(out)':(-2,2)}
