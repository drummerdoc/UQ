#   A class for parallelizing over independent likelihood calls
#   This is different from the map module in multiprocessing because
#   the procedure that evaluates the likelihood call stays live while
#   the rest of the emcee sampler is doing stuff.

#   To the calling program it looks similar to the asynchronous map
#   parallelism in the emcee hammer code.  But it is based on the
#   queue communication mechanism instead.

#   LivingMap.py

#   This module implements the LivingMap class.  The interface is three
#   procedures.  More details below:

#      _init( n_procs, target_function, pass_through_args) -- the constructor

#      Par_comp( job_arg_list)   -- farm the jobs to the processes, like map

#      stop()         -- end all the processes


from multiprocessing import Process, Lock, Queue
import time

class LivingMap:

   def __init__( self, n_procs, target_function, pass_through_args):
      """start n_proc processes.  Each one is the target_function with its
      pass_through_args.  Maintain a list of the processes.  Create two
      queues, one for passing work to the target_functions, and one for
      getting results back.  Pass a lock to the target_functions so they
      can initialize themselves in serial."""

      self._target_init_lock    = Lock()
      self._target_input_queue  = Queue()
      self._target_output_queue = Queue()
      self._proc_list           = []

      for proc in range(n_procs):
         proc = Process( target = target_function,
                         args   = (self._target_init_lock,
                                   self._target_input_queue,
                                   self._target_output_queue,
                                   pass_through_args,))
         self._proc_list.append(proc)

      for proc in self._proc_list:
         proc.start()

      return
      
   def Par_comp( self, task_arg_list):   # Parallel computing
      """Use the active processes to do the jobs in task_arg_list.
      Each item in the list jtask_arg_list is a task_arg.  Pass each
      task_arg to a process.  Create a list of values, one for each
      task_arg.  Return this list."""
      
#         pass out the tasks to the processes
   
      task_id = 0
      for task_arg in task_arg_list:
         task_description = tuple( [task_id, task_arg])
         self._target_input_queue.put( task_description)
         task_id = task_id + 1

#         collect the results into a list

      number_of_tasks = len( task_arg_list)
      task_value_list = [0 for i in range(number_of_tasks)]   # just to get a list of the right length
      for i in range(number_of_tasks):
         task, task_value      = self._target_output_queue.get()
         task_value_list[task] = task_value
      return task_value_list

      


   def stop(self):
      """Send the stop signal to all processes, then join them all"""
      
      dummy = 0                              # serves no purpose except to be a variable
      job_info = tuple( [-1, dummy])         # tell the processes to stop
      for proc in self._proc_list:
         self._target_input_queue.put( job_info)
      
      for proc in self._proc_list:           # join the processes, which puts them out of existence.
         proc.join()
      return



