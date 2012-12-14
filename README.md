evolutionary-circuits
=====================

Automatically generates analog circuits using evolutionary algorithms.

Requires ngspice, python and pypy (Yes, both of them). Also probably only works on a linux OS.

Interesting stuff happens in cgp.py, circuits.py is used to communicate with
ngspice, plotting.py is used to implement plotting with matplotlib when running
with pypy. getch.py gets a single character from output.

Work is currently in progress and documentation is missing. This is
currently hard to use and requires a knowledge of how to use SPICE.

Simulation can be stopped with a keyboard interrupt. Progress is saved after
every generation and can be continued by just running the script again.

#Installing

##ngspice

Download [ngpspice](http://ngspice.sourceforge.net/) either the git version or the
current stable release, whichever you want. Git release often has performance
improvements and possibly new features, but it also comes with the best bugs. If
you are new to using SPICE it might be better to install latest stable
version(As of 2012-09-10 this is version 24).

Compile and install ngspice with following commands:

     $ mkdir release
     $ cd release
     $ ../configure --enable-xspice --disable-debug
     $ make
     $ sudo make install

#Usage

The program is used by saving simulation settings in separate file and
running it with command: "pypy cgp.py <filename>". See examples
folder for example scripts.

Using pypy is necessary. CPython raises PicklingError, because it can't 
pickle functions inside classes, which is required for multithreading.

##Simulation settings

Every name starting with underscore("\_") is ignored by cgp.py and they can be used for
internal functions in settings file.

Required settings are:
* title: Title of the simulation, this will be the name of the folder where
  output is placed.
* max\_parts: Maximum number of devices allowed.
* spice\_commands: SPICE simulation commands, can be string if there is only one
  simulation or list of string for more than one simulation.
* parts: A dictionary of available parts.
* fitness\_function: The goal function, that circuits are ranked against.

Optional settings are:
* common: Common constructs in SPICE simulation, eg. input resistance, output
  load...
* models: String of SPICE models.
* constraints: One function or a list of functions for constraints. Some entries
  in list can be None. Length of list must equal the number of SPICE
  simulations.
* population: Number of circuits in one generation. Default is 1000.
* nodes: Number of nodes where parts can be attached. Default is same as
  max\_parts.
* elitism: Number of circuits copied straight into a new generation. Default is
  1.
* mutation\_rate: Probability of mutations. Default is 0.7.
* crossover\_rate: Probability of crossovers. Default is 0.2.
* fitness\_weight: Weightings for fitness functions. Can be either numbers or
  functions of type "lambda x,\*\*kwargs", where x is simulation x-axis value,
  kwargs has keys: "extra\_value", "generation", possibly more as they are added.
  List of dictionaries. One list elements corresponds to one
  SPICE simulation. Dictionary keys are measurements('v(n2)','i(n1')...), and
  values are the weights the measurement is multiplied. Value can also be
  a function that takes input value and returns a number.
* extra\_value: List of tuples that are minimum and maximum of extra values that chromosome can hold. This is returned
  to fitness function as argument extra\_value. This can be used as example for
  transition voltage of an inverter.
* log\_file: Filename where simulation log is saved.
* plot\_titles: List of dictionaries of plot titles.
* plot\_yrange: List of dictionaries of plot Y-axis ranges. Can be useful if you
  turn the output plots into an animation, this avoids the rescaling of the
  axes.
* selection\_weight: Higher values make better performing circuits being picked
  more often and smaller values make selection more fair, 0.0 makes selection completely random.
  Default is 1.0 and values should be bigger than zero. Usable range us probably from 0.5 to 1.5. Don't set this too high or entropy in the gene pool is lost and genetic algorithm doesn't converge.
  You should only touch this if you know what you are doing.
* constraint\_weight: Weights for constraints, similar to fitness\_weight.
  Score added is percentage of values not passing times this score.
* max\_mutations: Maximum number of mutations without re-evaluating the fitness
  function.
* constraint\_free\_generations: Number of generations before constraint
  functions are enabled.
* gradual\_constraints: Apply constraint function scores gradually, default is
  True.
* constraint\_ramp: Number of generations before constraint scores are at
  maximum levels. Does nothing if gradual\_constraints if False.
* random\_circuits: Percentage of random circuits in new generation. Default is
  0.01(1%).
* plot\_every\_generation: True to plot every generation even if the best score
  is same or worse than the last generation. Useful is weights increase when
  current generation increases, doesn't need to be enabled for
  "gradual\_constraints" to work properly. Default is False.
* default\_scoring: Use default scoring(Squared error difference and weighting).
* custom\_scoring: User defined scoring function. None if none is given. Both
  user defined and default scoring can be active at the same time, in this case
  scores are summed. Function should take two arguments: "result" and
  \*\*kwargs. result is dictionary of measurements, each dictionary value is
  tuple of two lists: simulation x-axis and y-axis values. Example:
  {'v(n2)':[0.0,0.1,0.2],[1.0,1.1,1.2]}. Should return a score that has a type
  of float.
* timeout: SPICE simulation default timeout. This is raised automatically if
  default is too low.
* seed: Seed circuits to be added in the first generation. All of the devices
  need to be in the parts dictionary. Can be a list or string.
* seed\_copies: Copies of seed circuits.

First you need to decide what components you want to allow and add them to
the "parts" dictionary. The dictionary key is the name of component in SPICE, and the value
should be an another dictionary with contents:

    'nodes': Number of leads in the component
    'value': True/False, component has a value(Example: True for resistors and
    capacitors, their values are resistance and capacitance.
    False for example transistors)
    'min': Exponent of the components minimum value. Minimum value will be
    10^(min)
    'max': Exponent of the components maximum value.
    'spice': Extra options for spice. This is appended to the end of components
    description in spice. This is used for example transistors model.

Next you need SPICE simulation commands and a optional part models. Ground node is hard coded in SPICE to be 0.

Add "print <value>"(eg. "print v(n2)" or "print i(n2)") in the spice commands to get an output to the program. You can print more than one measurement from one spice simulation.

Other settings you should specify are the fitness function or a list of them if you have more than one simulation. These are the goals of the evolution, that circuits are scored against.

Definition of the fitness function is:

    def fitness\_function(a,b,**kwargs)

First argument is the input value of the measurement(eg. voltage, current,
decibels, phase...). Second is the name of the SPICE measurement(eg. v(n2),
vdb(n3), i(vin)). \*\*kwargs has the optional extra\_value of the chromosome and
the current generation.

For example a constant fitness function:

    def goal(v,k,**kwargs):
        """v is input, either voltage, frequency, time... depends on spice
        simulation. k is the name of node being measured"""
        return 1

Another fitness function: A spice simulation with more than one measurement and example use of \*\*kwargs. This
example has a current and voltage measurements:

    def goal(v,k,**kwargs):
        if k[0]=='v':   #Voltage
            return kwargs['extra_value'][0]    #1V
        elif k[0]=='i': #Current
            return 1e-3 if kwargs['generation]<10 else 0

Discontinuous fitness function that is 0 before time is 100ns and 5 after it:

    def goal(t,k,**kwargs):
        #t is now time
        return 0 if t<=100e-9 else 5

Fitness weights are either constants that scores are multiplied with, or
functions that individual values of spice simulation are weighted with, if you want
to for example weight some part of the results higher than other parts. They
should be in list of dictionaries where number of elements in list is same as
number of spice simulations. The nth element is used with nth spice simulation. Keys
of the dictionaries are names of measurements(eg. 'v(n2)') and values are
weights.

For example a constants weights with 4 simulations with second one having two
measurements:

    fitness_weight=[{'v(n2)':1},{'v(n2)':1,'i(vin)':50},{'v(n2)':1},{'v(n2)':0.05}]


See examples folder for examples. Due to the rapid development examples might not
always work, but I try my best to keep them up to date.
