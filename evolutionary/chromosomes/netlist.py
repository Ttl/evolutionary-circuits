from common import *

class Device:
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

def random_element(parts,node_list,fixed_node=None):
    #Return random circuit element from parts list
    name = random.choice(parts.keys())
    part = parts[name]
    spice_line = []
    if 'value' in part.keys():
        minval,maxval = part['value'][:2]
        spice_line.append(log_dist(minval,maxval))
    nodes = [random.choice(node_list) for i in xrange(part['nodes'])]
    while same(nodes):
        nodes = [random.choice(node_list) for i in xrange(part['nodes'])]
    if fixed_node!=None:
        nodes[0]=fixed_node
        random.shuffle(nodes)
    if 'model' in part:
        if type(part['model'])!=str:
            spice_line.append(random.choice(part['model']))
        else:
            spice_line.append(part['model'])
    if 'cost' in part:
        cost = part['cost']
    else:
        cost = 0
    return Device(name,nodes,cost,spice_line)

def mutate_value(element,parts,rel_amount=None):
    i = random.randint(0,len(element.values)-1)
    val = element.values[i]
    name = element.spice_name
    if rel_amount==None:
        try:
            val[i] = log_dist(parts[name]['value'][0],parts[name]['value'][1])
        except:
            return element
    else:
        try:
            temp = val[i]*(2*random.random-1)*rel_amount
            if parts[name]['value'][0]<=temp<=parts[name]['value'][1]:
                val[i] = temp
        except:
            return element
    try:
        cost = parts[element.spice_name]['cost']
    except KeyError:
        cost = 0
    return Device(element.spice_name,element.nodes,cost,val)

class Chromosome:
    """Class that contains one circuit and all of it's parameters"""
    def __init__(self,max_parts,parts_list,nodes,extra_value=None):
        #Maximum number of components in circuit
        self.max_parts = max_parts
        #List of nodes
        self.nodes = nodes
        self.parts_list = parts_list
        #Generates randomly a circuit
        self.elements = [random_element(self.parts_list,self.nodes) for i in xrange(random.randint(1,int(0.75*max_parts)))]
        self.extra_range = extra_value
        if extra_value!=None:
            self.extra_value = [random.uniform(*i) for i in self.extra_range]
        else:
            self.extra_value = None

    def __repr__(self):
        return '\n'.join(map(str,self.elements))

    def get_connected_node(self):
        """Randomly returns one connected node"""
        if len(self.elements)>0:
            device = random.choice(self.elements)
            return random.choice(device.nodes)
        else:
            return 'n1'

    def value_bounds(self):
        bounds = []
        for e in self.elements:
            if 'value' in self.parts_list[e.spice_name]:
                bounds.append(self.parts_list[e.spice_name]['value'][:2])
        if self.extra_value != None:
            bounds.extend(self.extra_range)
        return bounds

    def get_values(self):
        values = []
        for e in self.elements:
            if 'value' in self.parts_list[e.spice_name]:
                values.append(e.values[0][0])
        if self.extra_value != None:
            values.extend(self.extra_value)
        return values

    def set_values(self, values):
        i = 0
        for e in self.elements:
            if 'value' in self.parts_list[e.spice_name]:
                e.values = ([values[i]],)
                i += 1
        if self.extra_value != None:
            self.extra_value = values[i:]
        return None

    def crossover(self, other):
        #if len(self.instructions)<len(other.instructions):
        #    return other.crossover(self)
        r = random.randint(0,1)
        l = max(len(self.elements),len(other.elements))
        r1 = random.randint(0,l)
        r2 = random.randint(0,l)
        if r1>r2:
            r1,r2=r2,r1
        if r==0:
            #Two point crossover
            self.elements = self.elements[:r1]+other.elements[r1:r2]+self.elements[r2:]
            self.elements = self.elements[:self.max_parts]
        else:
            #Single point crossover
            self.elements = self.elements[:r1]+other.elements[r1:]
            self.elements = self.elements[:self.max_parts]

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


    def spice(self,options):
        """Generate the input to SPICE"""
        program = options+'\n'
        for i in self.elements:
            program+=str(i)+'\n'
        return program


def random_circuit(parts, inst_limit, sigma, inputs, outputs, special_nodes, special_node_prob, extra_value=None):
    """Generates a random circuit.
    parts - dictionary of available devices.
    inst_limit - maximum number of nodes
    sigma - standard deviation of nodes.
    inputs - input nodes.
    outputs - output nodes.
    special_nodes - power supplies and other useful but not necessary nodes.
    special_node_prob - Probability of having a special node in single instruction,
        can be a list or single number.
    """
    #Add the ground
    special = special_nodes[:]
    if '0' not in special:
        special.append('0')
    special.extend(range(1,inst_limit))
    if max(len(inputs),len(outputs)) > inst_limit:
        raise ValueError("Number of allowed nodes is too small.")
    special = special + inputs + outputs
    c = Chromosome(inst_limit,parts,special,extra_value=extra_value)

    #Check for input and outputs
    has_input = False
    has_output = False
    for e in c.elements:
        if any(i in e.nodes for i in inputs):
            has_input = True
        if any(o in e.nodes for o in outputs):
            has_output = True
    if not has_input:
        c.elements[0].nodes[0] = random.choice(inputs)
        random.shuffle(c.elements[0].nodes)
    if not has_output:
        c.elements[-1].nodes[0] = random.choice(outputs)
        random.shuffle(c.elements[-1].nodes)
    return c


def parse_circuit(circuit, inst_limit, parts, sigma, inputs, outputs, special_nodes, special_node_prob, extra_value=None):
    devices = []
    special = special_nodes[:]
    if '0' not in special:
        special.append('0')
    if max(len(inputs),len(outputs)) > inst_limit:
        raise ValueError("Number of allowed nodes is too small.")
    special = special + inputs + outputs
    #Devices starting with same substring are sorted longest
    #first to check longest possible device names first
    sorted_dev = sorted(parts.keys(),reverse=True)
    nodes = {}
    len_nodes = 1
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
                line = line.split()
                d_nodes = line[1:parts[dev]['nodes']+1]
                for node in d_nodes:
                    if node not in special and node not in nodes:
                        nodes[node] =len_nodes
                        len_nodes += 1
                d_nodes = [nodes[node] if node in nodes else node for node in d_nodes]
                d_spice = line[parts[dev]['nodes']+1:]
                for e in xrange(len(d_spice)):
                    #Correct types and change SPICE multipliers to bare numbers.
                    try:
                        d_spice[e] = multipliers(d_spice[e])
                    except:
                        #Not a number.
                        pass
                devices.append(Device(dev,d_nodes,0,d_spice))
                break

        else:
            #Device not found
            print "Couldn't find device in line {}:{}\nIgnoring this line".format(n,line)
    #def __init__(self,max_parts,parts_list,nodes,extra_value=None):
    special.extend(map(str,range(1,inst_limit)))
    circuit = Chromosome(inst_limit, parts, special, extra_value)
    circuit.elements = devices
    return circuit

#parts = { 'R':{'value':(1,1e6),'nodes':2}, 'C':{'value':(1e-12,1e-3),'nodes':2}, 'Q':{'model':('2N3904','2N3906'),'kwvalues':{'w':(1e-7,1e-5),'l':(1e-7,1e-5)},'nodes':3} }
##r = Device('R',100,None,None,0)
##c = Device('C',1e-5,None,None,0)
##q = Device('Q',None,{'w':1,'l':2},'2N3904',0)
#c = random_circuit(parts, 10, 2, ['in1','in2'],['out'], ['vc','vd'], [0.1,0.1],extra_value=[(0,5)])
#print c
#c.mutate()
#print
#print c
#print c.value_bounds()
#d = c.get_values()
#print d
#c.set_values(d)
#print c
#print c.extra_value
