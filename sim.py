import scipy.stats
from numpy import matrix, zeros, ones
import numpy

# Represents a group of people
class bunch:  

  # This is the total of the goods they want to consume
  def total_goods_wanted(self):
    all_goods = matrix(zeros(len(goods))) # matrix([[0,0,0,0]])
    for person in self.people:
      all_goods += person.goods_wanted
    return all_goods

  # This is their estimate for what is needed to produce 
  # their target in their industry
  # (not that important -- not really used)
  def average_production_input(self):
    all_production_input = matrix(zeros(len(goods))) # matrix([[0,0,0,0]])
    for person in self.people:
      all_production_input += person.production_input
    return all_production_input / len(self.people)

  # Create a number of people
  def __init__(self, number):
    self.people = []
    for i in range(number):
      self.people += [person()]

# This is where we store the information about each industry
class Factory:

  # Create a factory
  def __init__(self, name, work_type, production_matrix, workers):
    self.name = name
    self.work_type = work_type # What proportions of work are nedded
    self.production_matrix = production_matrix
    self.workers = workers
    for w in workers.people:
      w.factory = self
    
    self.plan = None
  
  def production_output(self,production_input):
    # This is a function that determines the means of production
    # Can be the subject of discussion on the worker's council
    # and determining it is the key aspect of worker's self-management
    #
    # Here we fake it by pretending it is a simple linear
    # relationship between inputs and outputs.
    return (self.production_matrix * production_input.transpose())
  
  # This determines the needed inputs for the production
  # and is the result of self-management
  def production_input_total(self):
    if self.plan is not None:
      return self.plan
    return self.workers.average_production_input()
  
  # What all the workers want to consume
  def total_consumption(self):
    return self.workers.total_goods_wanted()
    
  def production_output_total(self):
    return self.production_output(self.production_input_total())
  
  # How many people need to work in this factory
  # (return the number of workers)  
  def units_of_work(self):
    return len(self.workers.people)

  # Do the maths to help compute prices...
  def update_to_price_matrix(self):
    # What we need to do the work
    raw_materials = self.production_input_total()    
    raw_labour = (self.work_type * self.units_of_work())
    all_raw = numpy.hstack((raw_materials,raw_labour)) 

    # Outputs from the work
    all_outputs = self.production_output_total()    
    all_outputs = numpy.vstack((all_outputs, zeros((len(goods),1))))
    
    m = ((all_outputs* 1.0) / sum(all_outputs.flat))  * all_raw
    for i in range(all_outputs.size):
      m[i,i] += -1.0 * all_outputs[i]

    # Now we sort out the consumption side of the matrix
    consume = self.total_consumption()
        
    #consume_all = numpy.hstack( (consume, zeros((1,4))) )
    consume_all = zeros((1,len(goods)+len(tasks)))
    work_all = numpy.hstack((zeros((1,len(tasks))), self.work_type))
    w = work_all.transpose() * consume_all
    
    total_work = self.units_of_work()
    for i in range(work_all.size):
       w[i,i] += work_all[0,i] 
   
    m = m + w
    
    return m

  # How much value is consumed and produced by this factory.
  def factory_value(self,values):
    # Inputs of production
    raw_materials = self.production_input_total()    
    all_raw = numpy.hstack((raw_materials,zeros((1,len(tasks))))) 
    value_inputs = values.transpose() * all_raw.transpose()
    value_inputs = value_inputs[0,0]

    # Outputs from the work
    all_outputs = self.production_output_total()    
    all_outputs = numpy.vstack((all_outputs, zeros((len(tasks),1))))
    value_outputs = values.transpose() * all_outputs
    value_outputs = value_outputs[0,0]
    
    value_added = value_outputs - value_inputs
    
    # Value added per unit labour
    raw_labour = (self.work_type * self.units_of_work())
    value_per_unit_work = value_added / sum(raw_labour.flat)
  
    return value_inputs, value_outputs, value_added, value_per_unit_work
 
# The heart of ParEcon to determine prices
class Economy:
  def __init__(self, factories):
    self.factories = factories
  
  # Goes through all factories and updates the price matrix  
  def price_matrix(self):
    econ_size = len(goods) + len(tasks)
    z = zeros((econ_size,econ_size))
    for f in self.factories:
      z += f.update_to_price_matrix()
    return z
  
  # Total goods produced by all industries
  def total_production(self):
    z = matrix(zeros(len(goods))).transpose()
    for f in self.factories:
      z += f.production_output_total()
    return z.transpose()
  
  # Total inputs needed by all industries
  def total_inputs(self):
    z = matrix(zeros(len(goods)))
    for f in self.factories:
      z += f.production_input_total()
    return z

  # Total goods needed by all people in the Economy
  def total_consumption(self):
    z = matrix(zeros(len(goods)))
    for f in self.factories:
      z += f.total_consumption()
    return z
  
  # Goods needed in the economy (industry + people + some slack)
  def total_needed(self):
    v = self.total_inputs() + self.total_consumption()
    v = v * 1.05 # Allow a 5% slack
    return v
  
  # Calculate the price of all goods
  def prices(self):
    econ = self.price_matrix()
    vals = numpy.hstack((zeros((1,len(goods))), ones((1,len(tasks)))))
    prices = numpy.linalg.solve(econ, vals.transpose())
    return prices
  
  # The value of all industries in the economy
  def value_of_industries(self):
    p = self.prices()
    print "prices: %s" % p.transpose()
    for fac in self.factories:
      (i_v, o_v, a_v, puw_v) = fac.factory_value(p)
      print "%s\t%s\t%s\t%s\t%s" % (fac.name, i_v, o_v, a_v, puw_v)

# Instanciate a first economy

# Define the goods and tasks in the Economy
goods = ["Blue stuff", "Red stuff", "Purple stuff", "Colourless stuff"]
tasks = ["Extract Blue", "Extract Red", "Mix to purple", "Scrub to colourless"]

PRICES = 8 # how many things we value: goods + tasks

# Store the production matrices (pM is 4 x 8)
#
# pI: input production vector [ goods, tasks] -- 8 rows
# pO: outpur production [ goods ] -- 4 rows
# pI x pM = pO
production = {}

# Unit production matrices
# 
# Describe in our model the relationship between inputs and production outputs
# (In reality this is the product of self management.)
#
production["Extract Blue"] = matrix([[10, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0] ])

production["Extract Red"] = matrix([[0, 0, 0, 0],
                                     [0, 10, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])

production["Mix"] = matrix([[0,0,0,0],
                            [0,0,0,0],
                            [1,1,0,0],
                            [0,0,0,0] ])

production["Scrub"] = matrix([[0,0,0,0],
                              [0,0,0,0],
                              [0,0,0,0],
                              [1,1,1,0] ])


# Define a person and their "needs"
class person:
  def __init__(self):    
    # These are determined by a utility function
    # (fix them for this silly example)
    self.goods_wanted = matrix([[5, 5, 5, 5]])
    self.goods_received = matrix([[0, 0, 0, 0]]).transpose()
    
    # This is the initial 'guess' of this person regarding what is needed
    # for their workplace to meet their production output.
    self.production_input = matrix([[50, 0, 0, 0]])

    # Where do I work?
    self.factory = None # determine this by bunch

# Factory syndicates

blue_bunch = bunch(10)
red_bunch = bunch(10)
purple_bunch = bunch(10)
shitbleach_bunch = bunch(10)

blue_factory = Factory("Blue & co", 
                        matrix([1,0,0,0]), 
                        production["Extract Blue"], 
                        blue_bunch)

blue_factory.plan = matrix([[50, 0, 0, 0]])


red_factory = Factory("Red'r'us", 
                        matrix([0,1,0,0]), 
                        production["Extract Red"], 
                        red_bunch)
                        
red_factory.plan = matrix([[0, 50, 0, 0]])

purple_factory = Factory("McPurple", 
                        matrix([0,0,1,0]), 
                        production["Mix"], 
                        purple_bunch)

purple_factory.plan = matrix([[110,110,0,0]])

bleach_factory = Factory("Whitewash & Son", 
                        matrix([0,0,0,1]), 
                        production["Scrub"], 
                        shitbleach_bunch)

bleach_factory.plan = matrix([[105,105,0,0]])

# The whole economy:
# -----------------

# Make an economy out of all the factories
myFirstEconomy = Economy([blue_factory, red_factory, purple_factory, bleach_factory])

# Compute all the prices for this economy
econ = myFirstEconomy.price_matrix()

# Print how much each factory is getting in and getting out
myFirstEconomy.value_of_industries()

# Print out the needed goods and produced goods (to detect deficit)
print "PRODUCTION: %s" % myFirstEconomy.total_production()
print "NEEDED: %s" % myFirstEconomy.total_needed()

# Determine if the budget of workers satisfies their needs
worker_budget = matrix([[5,5,5,5,0,0,0,0]]) * myFirstEconomy.prices()
print "Worker budget: %s" % worker_budget
