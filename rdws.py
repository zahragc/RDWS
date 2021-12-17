from functions import func, bdf, estimate
from env import IaaS , Workload
from env.dax_parser import parseDAX
from env.workflow import Workflow
from env.task import TaskStatus

from operator import attrgetter
import seaborn as sbn
import datetime
import random
import simpy
import math
import weakref


sbn.set_style("darkgrid", {'axes.grid' : True, 'axes.edgecolor':'black'})

import os
import torch
import numpy

def setRandSeed(seed):
  os.environ["PYTHONHASHSEED"] = str(seed);
  torch.manual_seed(seed);
  numpy.random.seed(seed); 
  random.seed(seed);

def runEnv(wf_path, taskScheduler, seed, constant_df=0, constant_bf=0, arrival_rate = 1/60,
           merge=False, wf_number=1, debug=False):
  
  global remained_tasks
  global workload_finished
  global running
  running = True;
  remained_tasks = 0;
  workload_finished = False; 
  
  wf_arrival_rate = arrival_rate; #workflows/secs
  boot_time = 97; #sec
  cycle_time = 3600; #sec
  bandwidth = 20000000#(2**20); # Byte   #20 MBps
  
  sim = simpy.Environment();
  workflow_submit_pipe = simpy.Store(sim);
  task_finished_announce_pipe     = simpy.Store(sim);
  vm_release_announce_pipe     = simpy.Store(sim);
  ready_queue_key = simpy.Resource(sim, 1);
  ready_task_counter = simpy.Container(sim, init=0);
  
  all_task_num = 0
  finished_wfs = [];
  workflow_pool = [];
  released_vms_info = [];
  tasks_ready_queue = [];
  all_vms = [];
  
  
  
  iaas = IaaS(sim, bandwidth,debug=False)
  iaas.addVirtualMachineType("m3_medium"  ,3    , 0.067, boot_time, cycle_time); 
  iaas.addVirtualMachineType("m4_large"   ,6.5  , 0.126, boot_time, cycle_time); 
  iaas.addVirtualMachineType("m3_xlarge"  ,13   , 0.266, boot_time, cycle_time);
  iaas.addVirtualMachineType("m4_2xlarge" ,26   , 0.504, boot_time, cycle_time); 
  iaas.addVirtualMachineType("m4_4xlarge" ,53.5 , 1.008, boot_time, cycle_time);
  iaas.addVirtualMachineType("m4_10xlarge",124.5, 2.520, boot_time, cycle_time); 
  
  
  fastest_vm_type = max(iaas.vm_types_list, key=attrgetter('mips'));
  cheapest_vm_type = min(iaas.vm_types_list, key= lambda v: v.cycle_price);
  setRandSeed(seed*5)
  workload = Workload(sim, workflow_submit_pipe, wf_path, wf_arrival_rate, max_wf_number = wf_number, debug=0)
  
  
    
  def __poolingProcess():
    global workload_finished 
    global remained_tasks
    while running and not workload_finished:
      dax_path = yield workflow_submit_pipe.get();

      if(dax_path == "end"):
          workload_finished = True;
          return;

      #Parse DAX and make a workflow
      tasks, files = parseDAX(dax_path, merge = False);
      
      wf = Workflow(tasks, path=dax_path, submit_time= sim.now);
      for task in wf.tasks:
          task.status = TaskStatus.pool;
          task.rank_trans = estimate.maxParentInputTransferTime(task, fastest_vm_type)
          task.rank_exe = estimate.exeTime(task, fastest_vm_type);
          wf.remained_length += task.rank_exe
      func.setUpwardRank(wf.exit_task, 0);
      func.setDownwardRank(wf.entry_task, 0);
      
#       for t in wf.tasks:
#         print(t.id, t.uprank, t.downrank)
      setRandSeed(seed+int(sim.now))
      bdf.createDeadline(wf, fastest_vm_type, constant_df=constant_df)
      bdf.createBudget(wf, cheapest_vm_type, constant_bf=constant_bf)
    
      
      workflow_pool.append(wf);
      remained_tasks += (len(wf.tasks)-2);
      wf.entry_task.status = TaskStatus.done;
      wf.entry_task.start_time = sim.now;
      wf.entry_task.finish_time = sim.now;
      
      #             if debug:
      print("[{:.2f} - {:10s}] {} (id: {}, deadline: {:.2f}, budget: {:.2f}, df: {:.2f}, bf: {:.2f}) is saved in the pool.\n # current Wf:{} # total Wf:{}"
            .format(sim.now, "Pool" ,dax_path, wf.id, wf.deadline, wf.budget, 
                    wf.deadline_factor, wf.budget_factor,
                    len(workflow_pool), len(workflow_pool) + len(finished_wfs)));

      wf.entry_task.status = TaskStatus.done;
      wf.entry_task.finish_time = sim.now;
      
      
      __addToReadyQueue(wf.entry_task.succ);
      yield ready_task_counter.put(1);

#       yield task_finished_announce_pipe.put(wf.entry_task);

  
  def __addToReadyQueue(task_list):
      for t in task_list:
          t.status = TaskStatus.ready;
          t.ready_time = sim.now;
      request_key = ready_queue_key.request();
      tasks_ready_queue.extend(task_list);
      ready_queue_key.release(request_key);

      if debug:
          print("[{:.2f} - {:10s}] {} tasks are added to ready queue. queue size: {}."
                .format(sim.now, "ReadyQueue", len(task_list), len(tasks_ready_queue)))    

  def __queueingProcess():
    while running:
      finished_task = yield task_finished_announce_pipe.get();
      finished_task.status = TaskStatus.done
      wf = finished_task.workflow
      wf.finished_tasks.append(finished_task);
        
      ready_tasks = [];
      for child in finished_task.succ:
        if child.isReadyToSch():
#             print(child.id)
            if child!=wf.exit_task:
                if merge: func.mergeOnFly(child);
                ready_tasks.append(child);
            else:
#                 print("///////////////////")
                wf.exit_task.status = TaskStatus.done;
                wf.exit_task.start_time = sim.now;
                wf.exit_task.finish_time = sim.now;
                wf.makespan = wf.exit_task.finish_time - wf.submit_time
                finished_wfs.append(wf)
                workflow_pool.remove(wf)
                print("[{:.2f} - {:10s}] Workflow {} is finished.".format(sim.now, "Finished", wf.id ));
                print("Deadline: {} Makespan: {}, Budget: {}, Cost: {}".format(wf.deadline, wf.makespan, wf.budget, wf.cost));
                print("*"*40)
                
      yield sim.timeout(0.2)
      if ready_tasks:
          __addToReadyQueue(ready_tasks);
          yield ready_task_counter.put(1);
  

  def threeDeadline(tasks_list, fastest_type, now_time):
    for task in tasks_list:
        task_len = estimate.maxParentInputTransferTime(task, fastest_type)+estimate.exeTime(task, fastest_type)        
        remained_deadline = task.workflow.deadline + task.workflow.submit_time - now_time;
        if remained_deadline < 0 : remained_deadline = 0;
#         task.deadline = task_len * remained_deadline / (task.uprank + task_len)
        task.deadline = ((task_len  * remained_deadline)
                         /(estimate.maxParentInputTransferTime(task, fastest_type)+ task.uprank))
        


  def estimateRunTimeCost(task_list, vm_list, vm_types_list, now_time, changed_vm = None, new_vm = None):
    for task in task_list:
        if changed_vm or new_vm:
            v = changed_vm if changed_vm else new_vm
            a = estimate.exeTime(task, v) + estimate.maxParentInputTransferTime(task, v)+v.waitingTime()
            b = estimate.exeCost(task, v) 
#             task.vref_time_cost.update({weakref.ref(v): a})
            task.vref_time_cost.update({v: [a, b]})
        else:
            task.vref_time_cost = {}
            for v in vm_list + vm_types_list:
                a = estimate.exeTime(task, v) + estimate.maxParentInputTransferTime(task, v)+v.waitingTime()
                if a<0:
                  print("$"*80)
                  print(estimate.exeTime(task, v),estimate.maxParentInputTransferTime(task, v),v.waitingTime())
                b = estimate.exeCost(task, v) 
#                 task.vref_time_cost.update({weakref.ref(v): a}) 
                task.vref_time_cost.update({v: [a, b]})
                
       
        task.vref_time_cost = dict(sorted(task.vref_time_cost.items(), key=lambda item: item[1][0]))
        task.fast_run = list(task.vref_time_cost.values())[0][0]

  
  def prioritizeTasks(task_list):
#       def slackTime(t): 
#         waiting_time = now_time - task.ready_time;
#         return (task.deadline - waiting_time) - fast_run;

    task_list.sort(key=lambda t: t.deadline - t.fast_run);
  
  def __releasingProcess():
    while running:
      vm = yield vm_release_announce_pipe.get();
      iaas.releaseVirtualMachine(vm);
      all_vms.remove(vm)
      released_vms_info.append(vm);
      if debug:
          print("[{:.2f} - {:10s}] {} virtual machine is released. start time: {}. VM number: {}"
              .format(sim.now, "Releaser", vm.id, vm.start_time, len(all_vms)));
                
      
  def __schedulingProcess():
    global workload_finished 
    global remained_tasks
    while running:
      yield ready_task_counter.get(1);
      threeDeadline(tasks_ready_queue, fastest_vm_type, sim.now)
      changed_vm = None;
      new_vm = None;
      while len(tasks_ready_queue):
        estimateRunTimeCost(tasks_ready_queue, all_vms, iaas.vm_types_list, sim.now)
        
        # prioritizeTasks should be call after that the deadline distributed
        prioritizeTasks(tasks_ready_queue)
        
        choosed_task = tasks_ready_queue.pop(0);
        choosed_task.schedule_time = sim.now;
        remained_tasks -= 1;
        
        BFT, LFT = bdf.calBFT_LFT(choosed_task, sim.now, 
                                fast_run = list(choosed_task.vref_time_cost.values())[0][0],
                                slow_run = list(choosed_task.vref_time_cost.values())[-1][0])
        choosed_task.soft_deadline = BFT
        choosed_task.hard_deadline = LFT
        choosed_task.BFT = BFT
        choosed_task.LFT = LFT
        
        if debug:
            print("[{:.2f} - {:10s}] {} task choosed for scheduling. L:{}"
                .format(sim.now, "TaskChooser", choosed_task.id, choosed_task.length));
            
        all_task_num = 0
        for w in workflow_pool:
          all_task_num += (len(w.tasks)-2);
        
        vlist = list(choosed_task.vref_time_cost.keys()) + [];
        random.shuffle(vlist)
        vs = vlist[:6]+[]
        
        choosed_vm, q = taskScheduler(len(vlist)==6, choosed_task, vs, 
                                tasks_ready_queue, remained_tasks, all_task_num,
                             sim.now, remained_tasks==0 and workload_finished);
        
        if len(vlist)!=6:
          del vlist[:6]
          while True:
            if len(vlist)>4:
                vs.remove(choosed_vm)
                random_vm = random.choice(vs)
                vs = vlist[:4] + [choosed_vm] + [random_vm]
                random.shuffle(vs)
                choosed_vm, q = taskScheduler(False, choosed_task, vs, 
                                tasks_ready_queue, remained_tasks, all_task_num,
                             sim.now, remained_tasks==0 and workload_finished);
#                 print(choosed_task.id, "------------------2", vs)
                del vlist[:4]
            else:
#                 print(vs, 0)
                vs.remove(choosed_vm)
                random_vm = random.choice(vs)
                vs.remove(random_vm)
#                 print(vs, 1)
                while len(vlist)<4:
#                   print(vs, 2)
                  random_vm = random.choice(vs)
                  vlist.append(random_vm)
                  vs.remove(random_vm)
                  
                vs = vlist + [choosed_vm] + [random_vm]
                random.shuffle(vs)
                
                choosed_vm, q = taskScheduler(True, choosed_task, vs, 
                                tasks_ready_queue, remained_tasks, all_task_num,
                             sim.now, remained_tasks==0 and workload_finished);
                
#                 print(choosed_task.id, "--------------3", vs)
                break;
          
        choosed_task.workflow.cost += choosed_task.vref_time_cost[choosed_vm][1]
        choosed_task.workflow.remained_length -= choosed_task.rank_exe
        choosed_task.vref_time_cost = {};
        
        if(choosed_vm.isVMType()):
          if debug:
              print("[{:.2f} - {:10s}] A new VM with type {} is choosed (among {} options) for task {}."
                  .format(sim.now, "Scheduler", choosed_vm.name, 
                          len(iaas.vm_types_list)+len(all_vms), choosed_task.id));

          nvm = iaas.provideVirtualMachine(choosed_vm, off_idle=True);            
          nvm.task_finished_announce_pipe = task_finished_announce_pipe;
          nvm.vm_release_announce_pipe = vm_release_announce_pipe;
          changed_vm = None;
          new_vm = nvm;
          yield sim.process(nvm.submitTask(choosed_task));
          all_vms.append(nvm);

        else:
            changed_vm = choosed_vm;
            new_vm = None;
      
            if debug:
                print("[{:.2f} - {:10s}] {} VM with type {} is choosed for task {}."
                    .format(sim.now, "Scheduler", choosed_vm.id, choosed_vm.type.name, choosed_task.id));
            # print(choosed_vm.type.name, choosed_task.id,"o b", choosed_task.budget, "c",estimate.taskExeCost(choosed_task, choosed_vm), "used b:",choosed_task.workflow.used_budget);
            yield sim.process(choosed_vm.submitTask(choosed_task));
  def lastFunction():
    
#     if len(finished_wfs)==1:
#           wf = finished_wfs[0]
#           a =  1 if wf.cost<= wf.budget and wf.makespan<= wf.deadline else 0
#           return wf.makespan, wf.cost, wf.makespan/wf.deadline, wf.cost/wf.budget,  a

    total_time = []
    total_cost = []
    budget_meet = []
    deadline_meet = []
    both_meet = []
    for wf in finished_wfs:
        total_time.append(wf.makespan)
        total_cost.append( wf.cost)
        budget_meet.append(wf.cost/wf.budget)
        deadline_meet.append(wf.makespan/wf.deadline)
        
        if wf.cost<= wf.budget:
          pass #budget_meet+=1
        else:
          print("XXB", wf.budget, wf.cost, wf.budget - wf.cost)
          
        if wf.makespan<= wf.deadline:
          pass #deadline_meet+=1
        else:
          print("XXD", wf.deadline, wf.makespan, wf.deadline - wf.makespan)
          
        if wf.cost<= wf.budget and wf.makespan<= wf.deadline:
          both_meet.append(1)
        else:
          both_meet.append(0)
#     total_time /= len(finished_wfs)
#     total_cost /= len(finished_wfs)
#     budget_meet /=len(finished_wfs)
#     deadline_meet /= len(finished_wfs)
#     both_meet /= len(finished_wfs)
    return total_time, total_cost, deadline_meet, budget_meet, both_meet
  
  
  sim.process(__poolingProcess());
  sim.process(__schedulingProcess());
  sim.process(__queueingProcess());
  sim.process(__releasingProcess());
  sim.run();
  return lastFunction()