import os
import matplotlib.pyplot as plt
import numpy
import random
import math
from .workflow import Workflow


class Workload():
    counter = 0;
    def __init__(self, env, workflow_submit_pipe , wf_path, arrival_rate, max_wf_number=float('inf'), 
                 max_time = float('inf'), initial_delay=0, random_seed = -1, dax=None, debug=False):
        Workload.counter += 1;
        self.id = "wl" + str(Workload.counter);
        self.env = env;
        self.dax = dax;
        self.workflow_submit_pipe = workflow_submit_pipe;
        self.workflow_path = wf_path;
        self.arrival_rate = arrival_rate; # workflow per min
        self.initial_delay = initial_delay;
        self.max_time = max_time;
        self.max_wf_number = max_wf_number;
        self.debug = debug;
        self.finished = False;
        self.rand_seed = random.randint(0, 10000) if random_seed== -1 else random_seed;
        self.rand_state = numpy.random.RandomState(self.rand_seed);
        random.seed(self.rand_seed);


        self.__submitted_wf_number = 0;
        self.delays =[];
        self.workflows = [];
        self.__times =[];
                
        # starts the __run() method as a SimPy process
        env.process(self.__run());

    def showDelayHistogram(self):
        # Density histogram
        plt.hist(self.delays, color = 'lavender', edgecolor = 'black', bins = int(180/7));
        plt.title('Density Histogram of Workflows Arrival Delay');
        plt.xlabel('Delay');
        plt.ylabel('Workflows Number');
        plt.show();

    def showSubmitTimeHistogram(self):
        # Density histogram
        plt.hist(self.__times, color = 'lavender', edgecolor = 'black', bins = int(180/7));
        plt.title('Density Histogram of Workflows Arrival Time');
        plt.xlabel('Time');
        plt.ylabel('Workflows Number');
        plt.show();
        
    def __poissonDistInterval(self):
        # k = 0 and lambda = wf_per_second
        wf_per_interval = self.arrival_rate;
        return self.rand_state.poisson(1.0 / wf_per_interval);
        # return numpy.random.RandomState().exponential(1.0 / wf_per_interval);


    def __run(self):
        if self.dax:
            yield self.workflow_submit_pipe.put(self.dax);
            self.finished = True;
            yield self.workflow_submit_pipe.put("end");
            return;

        yield self.env.timeout(self.initial_delay);
        
        # submit workflows until reach the max_time or max_wf_number 
        while self.env.now < self.max_time and self.__submitted_wf_number < self.max_wf_number:
            random.seed(self.rand_seed+self.__submitted_wf_number);
            #Choose random DAX file from workflow_path

            if isinstance(self.workflow_path, list):
                wf_path = random.choice(self.workflow_path)
            elif os.path.isdir(self.workflow_path):            
                dax = random.choice(os.listdir(self.workflow_path));

                while dax[0] == '.':
                	dax = random.choice(os.listdir(self.workflow_path));

                wf_path = self.workflow_path + "/" + dax;
            else:
                yield self.workflow_submit_pipe.put(self.workflow_path);
                self.finished = True;
                yield self.workflow_submit_pipe.put("end");
                return;

            interval = self.__poissonDistInterval();
            yield self.env.timeout(interval);

            self.delays.append(interval);
            self.__times.append(self.env.now);

            
            if self.debug:
                print("[{:.2f} - {:10s}] workflow {} ({}) submitted.".format(  
                    self.env.now, 'Workload', self.__submitted_wf_number, dax));

            self.workflows.append(wf_path);
            self.__submitted_wf_number += 1;

            yield self.workflow_submit_pipe.put(wf_path);     

            
            
        self.finished = True;
        yield self.workflow_submit_pipe.put("end");

    @staticmethod 
    def reset():
        Workflow.counter = 0;
        Workload.counter = 0;

    def __str__(self):
        return "Workload (id: {}, workflow_path: {}, arrival_rate: {})".format(
                    self.id, self.workflow_path, self.arrival_rate);

    def __repr__(self):
        return "{}".format(self.id);