from .task import TaskStatus
import weakref
import simpy
import math

class VirtualMachine:
    counter = 0;
    def __init__(self, env, vm_type, off_idle=False, debug = False):
        VirtualMachine.counter += 1;
        self.id = 'vm' + str(VirtualMachine.counter);
        self.num = VirtualMachine.counter + 0;
        self.debug = debug;
        self.env = env;
        self.off_idle = off_idle
        self.type = vm_type;
        self.provision_time = env.now;

        #--------------------------
        
        self.running      = False;
        self.executing_task = None;

        self.start_time   = 0;
        self.release_time = 0;
        self.done_tasks   = [];
        self.disk_items   = [];

        self.finish_time = 0;

        self.unfinished_tasks_number = 0;
        
        #The Store operates in a FIFO (first-in, first-out) order
        self.task_queue = simpy.Store(env);
        
        self.task_finished_announce_pipe     = None;
        self.vm_release_announce_pipe     = None;



    def start(self):
        self.env.process(self.__start());

        
    def __start(self):

        yield self.env.timeout(self.type.startup_delay);
        
        if(self.debug):
            print("[{:.2f} - {:10s}] Start.".format(self.env.now, self.id));
        self.start_time = self.env.now;
        self.running = True;

        if self.off_idle:
            self.env.process(self.__checkIdle());
        self.env.process(self.__cpu());

    def __checkIdle(self):
        while(self.running):
            yield self.env.timeout(self.type.cycle_time);
            # if(self.debug):

            if self.isIdle():
                # if(self.debug):
                #     print("[{:.2f} - {:10s}] Check if vm is idle. It is idle. Unfinished task number: {}"
                #         .format(self.env.now, self.id, self.unfinished_tasks_number));  
                
                self.vm_release_announce_pipe.put(self)
                self.release_time = self.env.now
                self.running = False;
            # else:
            #     if(self.debug):
            #         print("[{:.2f} - {:10s}] Check if vm is idle. It is not idle. Unfinished task number: {}"
            #             .format(self.env.now, self.id, self.unfinished_tasks_number));  
        
    def submitTask(self, task):
        self.unfinished_tasks_number += 1;
        self.estimateFinishTime(task);
        yield self.task_queue.put(task);
        task.vqueue_time = self.env.now;
        task.star_time_file_transferring = self.env.now;
        task.status = TaskStatus.wait;
        task.vm_ref = weakref.ref(self);

        if(self.debug):
            print("[{:.2f} - {:10s}] {} task is submitted to vm queue."
                .format(self.env.now, self.id, task.id));   

    def estimateFinishTime(self, task):
        vms = []
        transfer_time = []

        for ptask in task.pred:

            if not ptask.isEntryTask() and ptask.vm_ref().id != self.id:
                if ptask.vm_ref not in vms:
                    vms.append(ptask.vm_ref);

        for v in vms:
            total_size = 0;
            files = task.input_files + [];
            for file in files:
                if file in v().disk_items:        
                    total_size += file.size;
                    files.remove(file)

            transfer_time.append(v().type.transferTime4Size(total_size))

        trans_time = max(transfer_time) if transfer_time else 0; 
        task.estimate_waiting_time =  max(trans_time , self.waitingTime())
        task.estimate_transfer_time =  max(trans_time - self.waitingTime(), 0);

        task.estimate_finish_time = self.env.now + task.estimate_waiting_time + self.type.exeTime(task);
        self.finish_time = task.estimate_finish_time
        # print("waitingTime estimateFinishTime", task.id, task.estimate_waiting_time, task.estimate_finish_time)

    def __exeProcess(self, task):
        self.executing_task = task;
        task.start_time = self.env.now;
        task.status = TaskStatus.run;
        if(self.debug):
            print("[{:.2f} - {:10s}] {} task is start executing."
                .format(self.env.now, self.id, task.id));

        yield self.env.timeout(self.type.exeTime(task));
        
        task.finish_time = self.env.now;
        # task.status = TaskStatus.done;
        self.done_tasks.append(task);
        
        #make output files
        self.disk_items += task.output_files

        if(self.debug):
            print("[{:.2f} - {:10s}] {} task is finished."
                .format(self.env.now, self.id, task.id));
            
        self.unfinished_tasks_number -= 1;
        self.executing_task = None;
        self.task_finished_announce_pipe.put(task);

    def __cpu(self):
        while(self.running):
            task = yield self.task_queue.get();
            task.cpu_disposal_time = self.env.now;

            # I/O
            if task.estimate_transfer_time:
                yield self.env.timeout(task.estimate_transfer_time);

            self.disk_items += task.input_files;
            task.files_transfered = True;

            # CPU
            yield self.env.process(self.__exeProcess(task));

    def currentTaskRunningTime(self):
        return self.env.now - self.executing_task.start_time;

    def waitingTime(self):
        return max(self.finish_time - self.env.now, 0)

    def runningTime(self):
        if self.running:
            return self.env.now - self.start_time;
        return 0;

    def timeToStart(self):
        return 0 if self.running else self.type.startup_delay - (self.env.now - self.provision_time);

    def gap2EndCycle(self):
        return self.type.cycle_time - (self.finish_time%self.type.cycle_time)

        
    def isProvisionedVM(self):
        return True;
    
    def isVMType(self):
        return False;
            
    def isIdle(self):
        return self.running and  self.unfinished_tasks_number==0;


    @staticmethod 
    def reset():
        VirtualMachine.counter = 0;
        
    def __str__(self):
        return "VM (id: {}, type: {}, mips: {}, price: {})".format(
                    self.id, self.type.name, self.type.mips, self.type.cycle_price);
    
    def __repr__(self):
        return "{}".format(self.id);