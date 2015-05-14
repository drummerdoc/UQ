from multiprocessing import Process, Lock, Queue
import time


def f1( init_lock, input_queue, output_queue, init_args):
   """A procedure that prints, then waits for the stop signal"""
   
   init_lock.acquire()
   
#  The serial part of the setup

   setup_file_name = init_args[0]
   setup_file      = open( setup_file_name, 'r')
   file_line       = setup_file.readline()
   x               = float(file_line)
   
   init_lock.release()
   
   
   
#   The embarrassingly parallel part

   while (1):
   
      task, task_args = input_queue.get()   # wait for a task to come in
      
      if ( task >= 0) :                     # task>=0: do something
      
#    do the thing you've been asked to do
#    put the result (whatever kind of object it is) in task_value

         task_value  = [ task_args[0], x]
 
#    package the output message, containing the task id and computed value
         
         task_return = tuple( [task, task_value])
         output_queue.put( task_return)
      
#     The halt signal is a negative task id
      
      if ( task < 0):
         return

