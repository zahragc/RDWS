from .virtual_machine import VirtualMachine
from .virtual_machine_type import VirtualMachineType

class IaaS:
    counter = 0;    
    def __init__(self, env, average_bandwidth = 1, local_storage = True, debug = False):
        self.debug = debug;
        self.env = env;
        self.id = 'iaas' + str(IaaS.counter + 1);
        IaaS.counter += 1;
        self.average_bandwidth = average_bandwidth;
        self.local_storage = local_storage;
        #----------------------------------
        self.vm_types_list = [];
        self.provisioned_vm_list = [];
        self.released_vm_list = [];

    def provideVirtualMachine(self, vm_type, off_idle=False):
        vm =  VirtualMachine(self.env, vm_type, off_idle=off_idle, debug=self.debug);
        self.provisioned_vm_list.append(vm);
        
        if(self.debug):
            print("[{:.2f} - {:10s}] A virtual machine with type {} and id {} is provisioning ..."
                .format(self.env.now, "IaaS", vm_type.name, vm.id));
        
        vm.start();
        return vm;
                
    
    def releaseVirtualMachine(self, vm):
        # info = vm.release();
        # if (info):
        self.released_vm_list.append(vm);
        self.provisioned_vm_list.remove(vm);
        
            # if(self.debug):
            #     print("[{:.2f} - {:10s}] {} virtual machine is released. VMs number: {}"
            #         .format(self.env.now, "IaaS", vm.id, len(self.provisioned_vm_list)));
  
    def addVirtualMachineType(self, name, mips, cycle_price, boot_time, cycle_time):
        vm_type = VirtualMachineType(name, mips, cycle_price, self.average_bandwidth, cycle_time, boot_time);
        self.vm_types_list.append(vm_type);


    @staticmethod 
    def reset():
        VirtualMachine.reset();
        IaaS.counter = 0;
        
    def __str__(self):
        names = [];
        for vm_type in self.__vm_types:
            name.append(vm_type.name)
        return "IaaS => id: {}, boot_time: {}, bandwidth: {}, cycle_time: {}, vm_types: {}".format(
                    self.id, self.boot_time, self.bandwidth, self.cycle_time, names);
    
    def __repr__(self):
        return "{}".format(self.id);