import enum
class TaskStatus(enum.Enum):
    none  = 0;
    pool  = 1; 
    # 2: add to ready queue to be scheduled 
    ready = 2; 
    # 3: The task is waiting for the virtual machine. 
    # Waiting for the virtual machine to turn on, 
    # or waiting for its files to be transferred, 
    # or waiting for the other tasks to finish. 
    wait  = 3; 
    # 4: The task is running
    run   = 4; 
    done  = 5;
        
class Task():
    def __init__(self, num, length):
        self.id = "wf?-" + str(num);
        self.num = num;
        self.length = length;
        self.uprank = 0;
        self.downrank = 0;
        self.rank_exe = 0;
        self.rank_trans = 0;
        self.workflow = None;
        self.status = TaskStatus.none;
        self.depth = -1;
        self.height_len = -1;
        self.depth_len = -1;
        self.succ = [];
        self.pred = [];
        self.input_files = [];
        self.not_real_input_files = [];
        self.output_files = [];
        self.level = -1;
        self.deadline_cp = 0

        
        self.ready_time = 0;
        self.schedule_time = 0;
        self.vqueue_time = 0;
        self.cpu_disposal_time = 0;
        self.start_time = 0;
        self.finish_time = 0;

        self.store_in_temp = False;


        self.LP = -1; # Last Parent
        self.EST = -1;
        self.ECT = -1;
        self.EFT = -1;

        self.BFT = -1;
        self.LFT = -1;

        self.next_children = [];
        self.estimated_cost = -1;
        self.cost = 0;
        self.hard_deadline = -1;
        self.soft_deadline = -1;
        self.deadline = -1;
        self.budget = -1;

        self.vm_ref = None;
        self.files_transfered = False;
        self.star_time_file_transferring = 0;
        self.input_size = 0;
        self.estimate_waiting_time = 0;
        self.estimate_finish_time = 0;


        # Temp: for calculate Slack Time and scheduling
        self.fast_run = 0
        self.vref_time_cost ={} #  dictionary {v1:[2,0.5] v2: [23, 5] ......}


    def setWorkflow(self, wf):
        self.workflow = wf;
        self.id = str(wf.id) + "-" + str(self.num);    

    def isReadyToSch(self):
        for parent in self.pred:
            if(parent.status != TaskStatus.done):
                return False;
        return True;

    def isAllChildrenDone(self):
        for child in self.succ:
            if(child.status != TaskStatus.done):
                return False;
        return True;

    def isAllChildrenStoredInTemp(self):
        if self.succ[0].isExitTask():
            return True;
            
        for child in self.succ:
            if not child.store_in_temp:
                return False;
        return True;

    def isEntryTask(self):
        return len(self.pred)==0;

    def isExitTask(self):
        return len(self.succ)==0;

    def __str__(self):
        return "Task (id: {}, depth: {}, length: {},\n pred: {}, succ: {})".format(
                    self.id, self.depth, self.length, self.pred, self.succ);

    def __repr__(self):
        return "{}".format(self.id);
