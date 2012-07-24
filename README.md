evolutionary-circuits
=====================

Automatically generates analog circuits using evolutionary algorithms.

Requires ngspice. Also probably only works on a linux OS.

Interesting stuff happens in cgp.py, circuits.py is used to communicate with
ngspice, plot.py is not used by the program but it can be used to plot outputs of
ngspice. getch.py gets a single character from output.

Work is currently very much in progress and documentation is missing. This is
currently very hard to use and requires a knowledge of how to use SPICE.

Simulation can be stopped with a keyboard interrupt and it can be also resumed
without losing progress.

#Installing

##ngspice

Download [ngpspice](http://ngspice.sourceforge.net/) either the git version or the
current stable release, whichever you want.

Compile and install ngspice with following commands:

     $ mkdir release
     $ cd release
     $ ../configure --disable-debug
     $ make 2>&1 | tee make.log
     $ sudo make install

#Usage

The program is used by saving simulation settings in separate file and
running it with command: "python cgp.py <filename of settings>". See inverter.py
for example settings.

##Simulation settings

Every name starting with underscore("\_") is ignored and they can be used for
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
* fitness\_weight: List of dictionaries. One list elements corresponds to one
  SPICE simulation. Dictionary keys are measurements('v(n2)','i(n1')...), and
  values are the weights the measurement is multiplied. Value can also be
  a function that takes input value and returns a number.
* extra\_value: List of tuples that are minimum and maximum of extra values that chromosome can hold. This is returned
  to fitness function as argument extra\_value. This can be used as example for
  transition voltage of an inverter.
* log: Filename where simulation log is saved.
* plot\_titles: List of dictionaries of plot titles.
* plot\_yrange: List of dictionaries of plot Y-axis ranges. Can be useful if you
  turn the output plots into an animation, this avoids the rescaling of the
  axes.

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

Discontinous fitness function that is 0 before time is 100ns and 5 after it:

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


See the "inverter.py" for an example.
