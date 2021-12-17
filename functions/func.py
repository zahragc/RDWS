from functions import estimate
from operator import attrgetter
import types
import random

def setUpwardRank(task, rank): 
    if task.uprank < rank:
        task.uprank = rank;

    for ptask in task.pred:
        setUpwardRank(ptask, task.uprank + task.rank_trans + ptask.rank_exe);

def setDownwardRank(task, rank):
    if task.downrank < rank:
        task.downrank = rank;
    
    for ctask in task.succ:
        setDownwardRank(ctask, task.downrank + task.rank_exe + ctask.rank_trans);
        

def prioritizeTasks(task_list, all_vm, vm_types_list, now_time):
    for task in task_list:
        time = [];
        for v in vm_list:
            time.append(estimate.exeTime(task, v) + estimate.totalInputTransferTime(task, v)+v.waitingTime())


        fast_run = min(time);

   
    def slackTime(task, fast_run):
        waiting_time = now_time - task.ready_time;
        return (task.deadline - waiting_time) - fast_run;

    return tlist.sort(key=lambda x:slackTime(x));


    
def mergeOnFly(task):
    merge = True;
    while len(task.succ)==1 and task.succ[0].isExitTask()==False:
      for p in task.succ[0].pred:
        if p is not task and p.status is not TaskStatus.done:
          merge = False;
          break;
      if not merge:
        break;
      child = task.succ[0];
      task.num = str(task.num) + '+' + str(child.num)
      task.id = str(task.id) + '+' + str(child.id)
      task.length += child.length;

      for f in child.input_files:
          if f not in task.output_files and f not in task.input_files:
              task.input_files.append(f);

      for f in child.output_files:
          if f not in task.output_files:
              task.output_files.append(f);

      for t in child.succ:
          t.pred.remove(child);
          if task not in t.pred:
            t.pred.append(task);
          
      for t in child.pred:
          t.succ.remove(child);
          if task not in t.succ and t is not task:
              t.succ.append(task);

      task.succ += child.succ;
#       task.succ.remove(child);
      
      wf.tasks.remove(child);
      for t in wf.tasks:
        t.depth = 0;
      setTaskDepth(wf.entry_task, 0, 0);
      
      print("merge on fly", task.id)