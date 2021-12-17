from .task import TaskStatus

class Level:
    def __init__(self, number, tasks_list):
        self.num = number;
        self.tasks = tasks_list;
        self.deadline = 0;
        self.budget = 0;
        self.init_deadline = 0;

    def __str__(self):
        return "Level (num: {:<3d}, tasks: {}, deadline: {}, budget: {})".format(self.num, self.tasks, self.deadline, self.budget);
    
    def __repr__(self):
        return self.__str__();
        

class Workflow:
    counter = 0;
    def __init__(self, tasks, files = None, path = "", submit_time = 0, union=False):
        Workflow.counter += 1;
        self.id = 'wf' + str(Workflow.counter);
        self.path = path;
        self.user = "not important in this implementation!";

        self.fastest_exe_time = 0;
        self.deadline_factor = 0;
        self.cheapest_exe_cost = 0;
        self.budget_factor = 0;

        self.deadline = 0;
        self.budget = 0;
        self.used_budget = 0;
        self.submit_time = submit_time;        
        
        self.estimate_cost = 0;
        self.remained_length = 0;

        self.levels = [];

        self.finished_tasks = [];
        self.new_ready_tasks = 1;

        self.tasks = tasks;
        self.files = files;
                
        self.exit_task = None;
        self.entry_task = None;

        self.cost = 0;
        self.makespan = 0;


        for task in tasks:
            self.remained_length += task.length;
            task.setWorkflow(self);

            if task.isEntryTask():
                self.entry_task = task;
            elif task.isExitTask():
                self.exit_task = task;

            for input_file in task.input_files:
                input_file.consumer_tasks_id.append(task.id);

            for output_file in task.output_files:
                output_file.producer_task_id = task.id;

            # print(task.id, task.height_len)

    def getTaskNumber(self):
        return len(self.tasks);

    def getPoolTaskNumber(self):
        number = 0;
        for task in tasks:
            if(task.status == TaskStatus.none):
                number += 1;
        return number;
            
    @staticmethod 
    def reset():
        Workflow.counter = 0;
            
    def __str__(self):
        return "Workflow (id: {}, path: {})".format(self.id, self.path);
    
    def __repr__(self):
        return "{}".format(self.id);