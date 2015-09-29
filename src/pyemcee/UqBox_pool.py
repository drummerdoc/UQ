# Replacement for MPI pool to work with UqBox
# Parallel strategy where every MPI rank has it's own version
# of driver function that knows how to evaluate a likelyhood
# given a set of parameters, but function isn't pickleable
# Only parameters pickled, those get extracted and evaluated

# Ray Grout (ray.grout@nrel.gov)
# September 2015

from emcee.mpi_pool import MPIPool, _close_pool_message,\
                           _function_wrapper


class UqBoxPool(MPIPool):
    def __init__(self, MPI=None, debug=False):
        self.MPI = MPI
        self.function_arg = None
        super(UqBoxPool, self).__init__(comm=MPI.COMM_WORLD, debug=debug)

    def set_function(self, function):
        self.eval_function = function

    def set_function_arg(self, function_arg):
        self.function_arg = function_arg

    def wait(self):
        """
        If this isn't the master process, wait for instructions.

        """
        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        status = self.MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            if self.debug:
                print("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0,
                                  tag=self.MPI.ANY_TAG, status=status)
            if self.debug:
                print("Worker {0} got task {1} with tag {2}."
                      .format(self.rank, task, status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                if self.debug:
                    print("Worker {0} told to quit.".format(self.rank))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, _function_wrapper):
                self.function = task.function
                if self.debug:
                    print("Worker {0} replaced its task function: {1}."
                          .format(self.rank, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            # result = self.function(task)
            if self.function_arg:
                result = self.eval_function(self.function(task),
                                            self.function_arg)
            else:
                result = self.eval_function(self.function(task))
            if self.debug:
                print("Worker {0} sending answer {1} with tag {2}."
                      .format(self.rank, result, status.tag))
            self.comm.isend(result, dest=0, tag=status.tag)
