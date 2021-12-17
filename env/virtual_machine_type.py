class VirtualMachineType:
    def __init__(self, name, mips, price, bandwidth, cycle_time=3600, 
                 startup_delay=0, cpu_factor=1, net_factor=1):
        self.name = name;
        self.mips = mips;
        self.cycle_price = price;
        self.cycle_time = cycle_time;
        self.startup_delay = startup_delay;
        self.bandwidth = bandwidth;
        self.net_factor = net_factor; # error 
        self.cpu_factor = cpu_factor; # error 
        self.ram = "not important in this implementation!";
        self.disk_size = "not important in this implementation!";
        self.unfinished_tasks_number = 0


    def waitingTime(self):
        return self.startup_delay
        
    def exeTime(self, task):
        return (task.length/self.mips)*self.cpu_factor;

    def getLength(self, time):
        return  (time/self.cpu_factor)*self.mips;
    
    def transferTime(self, file):
        return (file.size/self.bandwidth)*self.net_factor; 

    def transferTime4Size(self, size):
        return (size/self.bandwidth)*self.net_factor; 


    def getSize(self, time):
    	return (time/self.net_factor)*self.bandwidth;
    
    def isVMType(self):
        return True;
    
    def isProvisionedVM(self):
        return False;
    
    def __str__(self):
        return "VM Type (name: {}, mips: {}, cycle_price: {})".format(
                        self.name, self.mips, self.cycle_price);
    
    def __repr__(self):
        return "VM Type {}".format(self.name);
    