from __future__ import(division, absolute_import, unicode_literals)

import numpy as np
from mpi4py import MPI


class _work_package(object):
    def __init__(self, task, idx):
        self.task = task
        self.idx = idx
        self.result = 0


class _no_work_message(object):
    def __repr__(self):
        return "<No work message>"


class _function_wrapper(object):
    def __init__(self, function):
        self.function = function


def _error_function(task, comm=None):
    raise RuntimeError("No valid function")


class H_MPIPool(object):

    # Three types of communicators we need to maintain:
    # the global communicator,
    # the subMaster communicator (including worker masters & global master),
    # and the worker communicators where every worker master is rank 0 and
    # it also contains the workers
    g_comm = 0
    g_rank = 0
    g_size = 0

    subMaster_comm = 0
    subMaster_rank = 0
    submaster_size = 0

    worker_comm = 0
    worker_rank = 0
    worker_size = 0

    subrankCounts = []
    br = []
    workerMasters = []
    sr = []

    work_count = 0

    argument = 0
    pass_communicator_to_function = False
    pass_argument_to_function = False
# Worker commands
    wcmd_STOP = 1
    wcmd_WORK = 2
    wcmd_NEWFCN = 3

    def __init__(self, comm):
        self.g_comm = comm
        self.g_rank = comm.Get_rank()
        self.g_size = comm.Get_size()

        self.function = _error_function

    def __del__(self):
        if(self.subMaster_comm):
            self.subMaster_comm.Free()
        if(self.worker_comm):
            self.worker_comm.Free()

    def is_master(self):
        if(self.g_rank == 0):
            return True
        else:
            return False

    def is_subMaster(self):
        if(self.subMaster_comm and not self.is_master()):
            return True
        else:
            return False

    def wait(self):
        done_work = False
        while (not done_work):
            # print "Global rank ", self.g_rank, " waiting "
            cmd = 0
            if(self.is_subMaster()):
                cmd = self.subMaster_comm.bcast(cmd, root=0)
            cmd = self.worker_comm.bcast(cmd, root=0)

            if(cmd == self.wcmd_WORK):
                self.work()
            elif(cmd == self.wcmd_STOP):
                done_work = True
            elif(cmd == self.wcmd_NEWFCN):
                F = _function_wrapper(_error_function)
                if(self.is_subMaster()):
                    F = self.subMaster_comm.bcast(F, root=0)
                F = self.worker_comm.bcast(F, root=0)
                self.function = F.function
            else:
                print "bad command on ", self.g_rank, ": ", cmd
        return

# Otherwise wait for the subMaster
        self.worker_comm.Barrier()

    def setSubrankCounts(self, nTopRanks=1, nSubRanks=1, distList=None):
        """ nTopRanks is the number of across which the
        function evaluations are spread, nSubRanks
        is the number of ranks for each function evaluation
        """

        # sr is the number of subranks for each top rank
        # first entry must be 1 for the master
        self.sr = [1]
        self.sr.extend(np.ones(nTopRanks, dtype=int)*nSubRanks)
        if(distList):
            self.sr = distList
        # self.sr = [1,2,4,2] # more interesting
        if (sum(self.sr) != self.g_size):
            raise RuntimeError("Distribution does not match rank count")

    def setupPool(self):
        # boundary ranks

        if(sum(self.sr) != self.g_size):
            # print "Need ", sum(self.sr), " ranks; have: ", self.g_size
            exit(-1)

        self.br = np.array([sum(self.sr[:i+1]) for i in range(len(self.sr))])

        colors = np.zeros(self.br.max())

        for i in range(self.br.max()):
            if(self.br[self.br <= i].size > 0):
                colors[i] = self.br[self.br <= i].max()

        self.workerMasters = np.unique(colors)
        # print "sr counts:", self.sr
        # print "br: ", self.br
        # print "colors: ", colors
        # print "workerMasters: ", self.workerMasters
        # Now pick of which color I am
        my_color = colors[self.g_comm.Get_rank()]

        # Setup colors and keys to split communicator for the worker masters
        # and grand master
        worldGroup = self.g_comm.Get_group()
        sr_root_group = worldGroup.Incl(list(self.workerMasters.astype(int)))
        self.subMaster_comm = MPI.Intracomm.Create(self.g_comm, sr_root_group)
        if(self.subMaster_comm):
            self.subMaster_size = self.subMaster_comm.Get_size()
            self.subMaster_rank = self.subMaster_comm.Get_rank()
        sr_root_group.Free()

        # Split global communicator again - note grand master is rank 0,
        # gets its own subcomm here
        self.worker_comm = self.g_comm.Split(my_color)

    def printHPool(self):
        sc_rank = self.worker_comm.Get_rank()
        sc_size = self.worker_comm.Get_size()-1
        if(self.is_subMaster()):
            sr_rank = self.worker_comm.Get_rank()
            sr_size = self.worker_comm.Get_size()-1
            print "Global rank {}/{}; local rank {}/{} (master; rank {}/{})"\
                  .format(self.g_rank, self.g_size, sc_rank,
                          sc_size, sr_rank, sr_size)
        elif (self.is_master()):
            sr_rank = self.worker_comm.Get_rank()
            sr_size = self.worker_comm.Get_size()-1
            print "Global rank {}/{}; local rank {}/{} (grand master; rank {}/{})"\
                  .format(self.g_rank, self.g_size,
                          sc_rank, sc_size, sr_rank, sr_size)
        else:
            print "Global rank {}/{}; local rank {}/{} (worker)"\
                  .format(self.g_rank, self.g_size, sc_rank, sc_size)

    def work(self, tasks=0):
        self.work_count += 1

        STEP = 0
        N = 0

        if(self.is_master()):
            STEP = self.subMaster_size-1
            N = len(tasks)
            # print "N, STEP = ", N, STEP
            result = np.zeros(N)

            # Send out 1 task per process up to N
            for i in range(0, N, STEP):
                ml = i
                mu = min(i+STEP+1, N)

                for p in range(1, STEP+1):
                    m = ml+p-1
                    if(m < mu):
                        WP = _work_package(tasks[m], m)
                        self.subMaster_comm.send(WP, p)
                        WP = self.subMaster_comm.recv(source=p)
                        result[WP.idx] = WP.result
            for p in range(1, STEP+1):
                self.subMaster_comm.send(_no_work_message(), p)
            return result
        else:
            have_work = True
            while (have_work):
                WP = _work_package(-1, -1)
                if(self.subMaster_comm):
                    WP = self.subMaster_comm.recv()
                WP = self.worker_comm.bcast(WP, 0)
                if not isinstance(WP, _no_work_message):
                    if(self.pass_communicator_to_function):
                        WP.result = self.function(WP.task,
                                                  comm=self.worker_comm)
                    elif(self.pass_argument_to_function):
                        WP.result = self.function(WP.task, pargs=self.argument)
                    elif(self.pass_communicator_to_function
                         and self.pass_argument_to_function):
                        WP.result = self.function(WP.task,
                                                  comm=self.worker.comm,
                                                  pargs=self.argument)
                    else:
                        WP.result = self.function(WP.task)

                    if(self.subMaster_comm):
                        self.subMaster_comm.isend(WP, 0)
                else:
                    have_work = False
            return

    def map(self, function, tasks):
        # Tell worker masters to work -
        # only grand master should call this; protect against if
        # the entire pool calls it

        if(not self.is_master()):
            self.wait()
            return

        if function is not self.function:
            # print "new function on master"

            self.function = function
            F = _function_wrapper(function)

            cmd = self.wcmd_NEWFCN
            self.subMaster_comm.bcast(cmd, root=0)
            self.subMaster_comm.bcast(F, root=0)

        cmd = self.wcmd_WORK
        self.subMaster_comm.bcast(cmd, root=0)

        R = self.work(tasks)
        return R

    def close(self):
        if(self.is_master()):
            print "grand master saying stop"
            cmd = self.wcmd_STOP
            self.subMaster_comm.bcast(cmd, root=0)

    def bcast(self, *args, **kwargs):
        return self.comm.bcast(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
