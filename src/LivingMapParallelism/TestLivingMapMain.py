#   A tester main program for the LivingMap module.

#   Test 1: create jobs and stop them

from multiprocessing import Process, Lock, Queue
import time
import livingmap
from   LivingMapTestRoutines import f1



if __name__ == "__main__":

   setup_file_name = "SerialSetupFile"
  
   setup_args = []
   setup_args.append(setup_file_name)
   proc_factory = livingmap.LivingMap( 4, f1, setup_args)
   
   task_list = []
   task_list.append( ["Dick",           20])
   task_list.append( ["Jane",           21])
   task_list.append( ["Sally",          22])
   task_list.append( ["Spot",           23])
   task_list.append( ["Burt",           24])
   task_list.append( ["Ernie",          25])
   task_list.append( ["Cookie Monster", 26])
   task_list.append( ["Elmo",           27])
   
   task_value_list = proc_factory.Par_comp( task_list)
   
   for task_value in task_value_list:
      print str(task_value)
   
   proc_factory.stop()




