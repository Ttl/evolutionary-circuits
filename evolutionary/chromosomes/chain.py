import random
import ast
from common import *

class Device:
    def __init__(self, name, value, kwvalues, model, cost):
        self.name = name
        self.value = value
        self.kwvalues = kwvalues
        self.model = model
        if cost!=None:
            self.cost = cost
        else:
            self.cost = 0

    def spice(self, nodes, device_number=None):
        kw = ''
        if self.kwvalues != None:
            for k in self.kwvalues.keys():
                kw += ' '+k+'='+str(self.kwvalues[k])
        if device_number==None:
            device_number = str(id(self))
        else:
            device_number = str(device_number)
        return self.name + device_number +' '+ ' '.join(map(str,nodes)) + ' '+ (str(self.value) if self.value!=None else '') +  (self.model if self.model!=None else '') + kw

    def __repr__(self):
        return self.spice('-',1)

    def mutatevalue(self, r):
        """Mutates device value. If r is 3-tuple third value is a random
        number distribution. Two first are lower and upper limits."""
        if len(r)==3:
            self.value = r[3](*r[:2])
        else:
            if r[0]>0:
                self.value = log_dist(*r)
            else:
                self.value = random.uniform(*r)

    def mutatekwvalue(self, r):
        """Mutates keyword values. Same logic as in value mutations."""
        kw = random.choice(self.kwvalues.keys())
        r = r[kw]
        if len(r)==3:
            self.kwvalues[kw] = r[3](*r[:2])
        else:
            if r[0]>0:
                self.kwvalues[kw] = log_dist(*r)
            else:
                self.kwvalues[kw] = random.uniform(*r)

def random_device(parts):
    """Generates a random device from parts list.
    "sigma" is the gaussian distribution standard deviation,
    which is used for generating connecting nodes."""
    name = random.choice(parts.keys())
    r = parts[name]
    kw,value,model = [None]*3
    cost = 0
    if 'kwvalues' in r.keys():
        kw = {i:value_dist(r['kwvalues'][i]) for i in r['kwvalues'].keys()}
    if 'value' in r.keys():
        value = value_dist(r['value'])
    if 'model' in r.keys():
        model = r['model'] if type(r['model'])==str else random.choice(r['model'])
    if 'cost' in r.keys():
        cost = r['cost']
    return Device(name, value, kw, model, cost)

def random_instruction(parts, sigma, special_nodes, special_node_prob, mu=0.5):
    """Generate random instruction with random device.
    "sigma" is the standard deviation of nodes."""
    d = random_device(parts)
    nodes = [int(round(random.gauss(mu,sigma))) for i in xrange(parts[d.name]['nodes'])]
    while same(nodes):
        nodes = [int(round(random.gauss(mu,sigma))) for i in xrange(parts[d.name]['nodes'])]
        #Sprinkle some special nodes
        if type(special_node_prob)==list:
            for i in xrange(len(nodes)):
                if random.random() < special_node_prob[i]:
                    nodes[i] = lst_random(special_nodes,special_node_prob)
        else:
            for i in xrange(len(nodes)):
                if random.random() < special_node_prob:
                    nodes[i] = random.choice(special_nodes)

    command = random.randint(0,1)
    return Instruction(command, d, sigma, nodes, special_nodes, special_node_prob)

class Instruction:
    def __init__(self, command, device, sigma, args, special_nodes, special_node_prob):
        self.commands = 2
        self.command = command
        self.device = device
        self.sigma = sigma
        self.args = args
        self.special_nodes = special_nodes
        self.special_node_prob = special_node_prob

    def __call__(self, current_node, device_number):
        #Transform relative nodes to absolute by adding current node
        nodes = [current_node + i if type(i)==int else i for i in self.args]
        #nodes = [i if i>=0 else 0 for i in nodes]
        if self.command == 0:
            #Connect
            return (self.device.spice(nodes,device_number),current_node)
        if self.command == 1:
            #Connect and move
            return (self.device.spice(nodes,device_number),current_node+1)
        #if self.command == 2:
        #    #Move to current_node + self.args
        #    return ('',current_node + self.args)
        raise Exception("Invalid instruction: {}".format(self.command))

    def __repr__(self):
        return self.__call__(0,1)[0]

    def mutate(self, parts):
        #Possible mutations
        m = [self.device.value!=None,self.device.kwvalues!=None,
                self.device.model!=None,True,self.command in (0,1)]
        m = [i for i,c in enumerate(m) if c==True]
        r = random.choice(m)
        if r == 0:
            #Change value
            self.device.mutatevalue(parts[self.device.name]['value'])
        elif r == 1:
            #Change kwvalue
            self.device.mutatekwvalue(parts[self.device.name]['kwvalues'])
        elif r == 2:
            #Change model
            models = parts[self.device.name]['model']
            if type(models)!=str and len(parts[self.device.name]['model'])>1:
                models = set(models)
                #Remove current model
                models.discard(self.device.model)
                self.device.model = random.choice(list(models))
        elif r == 3:
            #Change type of the instruction
            old = self.command
            new = set(range(self.commands))
            new.discard(old)
            self.command = random.choice(list(new))
        elif r == 4:
            #Change nodes
            self.args = [int(random.gauss(0,self.sigma)) for i in xrange(parts[self.device.name]['nodes'])]
            while same(self.args):
                self.args = [int(random.gauss(0,self.sigma)) for i in xrange(parts[self.device.name]['nodes'])]
                #Sprinkle some special nodes
                if type(self.special_node_prob)==list:
                    for i in xrange(len(self.args)):
                        if random.random() < self.special_node_prob[i]:
                            self.args[i] = lst_random(self.special_nodes,self.special_node_prob)
                else:
                    for i in xrange(len(self.args)):
                        if random.random() < self.special_node_prob:
                            self.args[i] = random.choice(self.special_nodes)


def random_circuit(parts, inst_limit, sigma, inputs, outputs, special_nodes, special_node_prob, extra_value=None):
    """Generates a random circuit.
    parts - dictionary of available devices.
    inst_limit - maximum number of instructions.
    sigma - standard deviation of nodes.
    inputs - input nodes.
    outputs - output nodes.
    special_nodes - power supplies and other useful but not necessary nodes.
    special_node_prob - Probability of having a special node in single instruction,
        can be a list or single number.
    """
    #Normalize probabilities and check that probabilities are valid
    special = special_nodes[:]
    if type(special_node_prob)==list:
        if not all(0<i<1 for i in special_node_prob):
            raise ValueError("Invalid probability in special node probabilities list. All probabilities need to be in the open interval (0,1).")
        if len(special_node_prob)!=len(special):
            raise ValueError("Special node lists are of different length.")
    else:
        if not 0<special_node_prob<1:
            raise ValueError("Invalid special node probability. Probability needs to be in the open interval (0,1).")
    #Add the ground
    if '0' not in special:
        special.append('0')
        if type(special_node_prob)==list:
            special_node_prob.append(0.1)
    if max(len(inputs),len(outputs)) > inst_limit:
        raise ValueError("Instruction limit is too small.")
    special = special + inputs + outputs
    if type(special_node_prob)==list:
        special_node_prob = special_node_prob + [0.1]*(len(inputs)+len(outputs))
    inst = [random_instruction(parts, sigma, special, special_node_prob) for i in xrange(random.randint(max(len(inputs),len(outputs)),inst_limit))]
    for e,i in enumerate(inputs):
        nodes = inst[e].args
        nodes[argmin(nodes)] = i
    for e,i in enumerate(outputs,1):
        nodes = inst[-e].args
        nodes[argmax(nodes)] = i
    return Circuit(inst, parts, inst_limit, (parts,sigma,special, special_node_prob), extra_value)

class Circuit:
    def __init__(self, inst, parts, inst_limit, inst_args, extra_value=None):
        #Everything necessary to make a new instruction
        self.inst_args = inst_args
        #List of instructions in the circuit
        self.instructions = inst
        #List of devices
        self.parts = parts
        #Max instructions
        self.inst_limit = inst_limit

        if extra_value!=None:
            self.extra_range = extra_value
            self.extra_value = [random.uniform(*i) for i in self.extra_range]
        else:
            self.extra_value = None

    def spice(self, commands):
        current_node = self.inst_limit
        program = ''
        for device_number,inst in enumerate(self.instructions,1):
            t = inst(current_node,device_number)
            program += t[0]+'\n'
            current_node = t[1]
        return commands+program

    def __repr__(self):
        return self.spice('')

    def mutate(self):
        #Available mutations
        m = [
                len(self.instructions)>0,
                len(self.instructions)>1,
                len(self.instructions)>1,
                len(self.instructions)<self.inst_limit,
                self.extra_value != None
            ]
        m = [i for i,c in enumerate(m) if c==True]
        r = random.choice(m)
        if r==0:
            #Single instruction mutation
            i = random.choice(self.instructions)
            i.mutate(self.parts)
        elif r==1:
            #Exchange two instructions
            i = random.randint(0,len(self.instructions)-1)
            c = random.randint(0,len(self.instructions)-1)
            self.instructions[i],self.instructions[c] = self.instructions[c],self.instructions[i]
        elif r==2:
            #Delete instruction
            i = random.randint(0,len(self.instructions)-1)
            del self.instructions[i]
        elif r==3:
            #Add instructions
            i = random_instruction(*self.inst_args)
            self.instructions.insert(random.randint(0,len(self.instructions)),i)
        elif r==4:
            #Change extra value
            self.extra_value = [random.uniform(*i) for i in self.extra_range]


    def crossover(self, other):
        #if len(self.instructions)<len(other.instructions):
        #    return other.crossover(self)
        r = random.randint(0,1)
        l = max(len(self.instructions),len(other.instructions))
        r1 = random.randint(0,l)
        r2 = random.randint(0,l)
        if r1>r2:
            r1,r2=r2,r1
        if r==0:
            #Two point crossover
            self.instructions = self.instructions[:r1]+other.instructions[r1:r2]+self.instructions[r2:]
            self.instructions = self.instructions[:self.inst_limit]
        else:
            #Single point crossover
            self.instructions = self.instructions[:r1]+other.instructions[r1:]
            self.instructions = self.instructions[:self.inst_limit]

    def value_bounds(self):
        """Return bounds of values for optimization."""
        bounds = []
        for ins in self.instructions:
            if ins.device.value != None:
                bounds.append(self.parts[ins.device.name]['value'])
            if ins.device.kwvalues != None:
                for kw in sorted(self.parts[ins.device.name]['kwvalues'].keys()):
                    bounds.append(self.parts[ins.device.name]['kwvalues'][kw])
        if self.extra_value != None:
            bounds.extend(self.extra_range)
        return bounds

    def get_values(self):
        values = []
        for ins in self.instructions:
            if ins.device.value != None:
                values.append(ins.device.value)
            if ins.device.kwvalues != None:
                for kw in sorted(self.parts[ins.device.name]['kwvalues'].keys()):
                    values.append(ins.device.kwvalues[kw])
        if self.extra_value != None:
            values.extend(self.extra_value)
        return values

    def set_values(self,values):
        i = 0
        for ins in self.instructions:
            if ins.device.value != None:
                ins.device.value = values[i]
                i += 1
            if ins.device.kwvalues != None:
                for kw in sorted(self.parts[ins.device.name]['kwvalues'].keys()):
                    ins.device.kwvalues[kw] = values[i]
                    i += 1
        if self.extra_value != None:
            self.extra_value = values[i:]
        return None

def parse_circuit(circuit, inst_limit, parts, sigma, inputs, outputs, special_nodes, special_node_prob, extra_value=None):
    """Converts netlist to chromosome format."""
    if '0' not in special_nodes:
        special_nodes.append('0')
    special = special_nodes + inputs + outputs
    #Devices starting with same substring are sorted longest
    #first to check longest possible device names first
    sorted_dev = sorted(parts.keys(),reverse=True)
    instructions = []
    #Table for converting circuit nodes to chromosome nodes
    nodes = {}
    len_nodes = 1
    current_node = 0
    for n,line in enumerate(circuit.splitlines()):
        if not line:
            #Ignores empty lines
            continue

        #Current device fields
        d_spice = []
        #Try all the devices
        for dev in sorted_dev:
            if line.startswith(dev):
                #Found matching device from parts list
                current_node += 1#Increase current node
                line = line.split()
                d_nodes = line[1:parts[dev]['nodes']+1]
                for node in d_nodes:
                    if node not in special and node not in nodes:
                        nodes[node] =len_nodes
                        len_nodes += 1

                d_spice = line[parts[dev]['nodes']+1:]
                for e in xrange(len(d_spice)):
                    #Correct types and change SPICE multipliers to bare numbers.
                    try:
                        d_spice[e] = multipliers(d_spice[e])
                    except:
                        #Not a number.
                        pass
                if 'value' in parts[dev]:
                    value = float(d_spice[0])
                    if not parts[dev]['value'][0]<=value<=parts[dev]['value'][1]:
                        raise ValueError("Value of component on line {} is out of bounds\n{}\nBounds defined in the parts dictionary are: {} to {}".format(n,' '.join(line),parts[dev]['value'][0],parts[dev]['value'][1]))
                    d_spice = d_spice[1:]
                else:
                    value = None
                if 'model' in parts[dev]:
                    model = d_spice[0]
                    d_spice = d_spice[1:]
                else:
                    model = None
                if 'kwvalues' in parts[dev]:
                    d_spice = ["'"+i[:i.index('=')]+"'"+i[i.index('='):] for i in d_spice]
                    kwvalues = ast.literal_eval('{'+', '.join(d_spice).replace('=',':')+'}')
                else:
                    kwvalues = None
                if 'cost' in parts[dev]:
                    cost = parts['cost']
                else:
                    cost = 0
                device = Device(dev, value, kwvalues, model, cost)
                node_temp = [nodes[node] - current_node + inst_limit if node in nodes else node for node in d_nodes]
                instructions.append(Instruction(1, device, sigma, node_temp, special, special_node_prob))
                break

        else:
            #Device not found
            print "Couldn't find device in line {}:{}\nIgnoring this line".format(n,line)
    if len(instructions) > inst_limit:
        raise ValueError("Maximum number of devices is too small for seed circuit.")
    return Circuit(instructions, parts, inst_limit, (parts, sigma, special, special_node_prob), extra_value)

#parts = { 'R':{'value':(1,1e6),'nodes':2}, 'C':{'value':(1e-12,1e-3),'nodes':2}, 'Q':{'model':('2N3904','2N3906'),'kwvalues':{'w':(1e-7,1e-5),'l':(1e-7,1e-5)},'nodes':3} }
#r = Device('R',100,None,None,0)
#c = Device('C',1e-5,None,None,0)
#q = Device('Q',None,{'w':1,'l':2},'2N3904',0)
#c = random_circuit(parts, 10, 2, ['in1','in2'],['out'], ['vc','vd'], [0.1,0.1])
#print c
#c.mutate()
#print c
#seed="""
#R1 n4 n3 1k
#R2 out n3 1k
#Q11 n5 n4 in1 2N3904
#Q12 n5 n4 in2 2N3904
#Q13 out n5 0 2N3904
#"""
#inputs = ['in1','in2']
#outputs = ['out']
#special = []
#print parse_circuit(seed, 10, parts, 2, inputs, outputs, special, 0.1)
