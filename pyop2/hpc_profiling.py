# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Profiling classes/functions."""

import numpy as np
from time import time
from contextlib import contextmanager
from configuration import configuration


class Timer(object):

    """Generic timer class.

    :param name: The name of the timer, used as unique identifier.
    :param timer: The timer function to use. Takes no parameters and returns
        the current time. Defaults to time.time.
    """

    _timers = {}
    _output_file = None
    _flp_ops = 0
    _gflops = 0.0

    def __new__(cls, name=None, timer=time):
        n = name or 'timer' + str(len(cls._timers))
        if n in cls._timers:
            return cls._timers[n]
        return super(Timer, cls).__new__(cls, name, timer)

    def __init__(self, name=None, timer=time):
        n = name or 'timer' + str(len(self._timers))
        if n in self._timers:
            return
        self._name = n
        self._timer = timer
        self._start = None
        self._timings = []
        self._volume = None
        self._timers[n] = self
        self._c_timings = []
        self._c_rand_timings = []
        self._output_file = None
        self._extra_param = -1
        self._only_kernel = False
        self._volume_mvbw = None
        self._volume_mbw = None

    def data_volume(self, vol, vol_mvbw, vol_mbw):
        self._volume = vol
        self._volume_mvbw = vol_mvbw
        self._volume_mbw = vol_mbw

    def c_time(self, c_time):
        """Time value from the kernel wrapper."""
        self._c_timings.append(c_time)

    def c_rand_time(self, c_rand_time):
        """Time value from the kernel wrapper in the randomized case."""
        self._c_rand_timings.append(c_rand_time)

    def papi_gflops(self, flp_ops, gflops):
        """Time value from the kernel wrapper."""
        Timer._flp_ops = max(Timer._flp_ops, flp_ops)
        Timer._gflops = max(Timer._gflops, gflops)

    def start(self):
        """Start the timer."""
        if self._name not in Timer._timers:
            self.reset()
            Timer._timers[self._name] = self
        self._start = self._timer()

    def stop(self):
        """Stop the timer."""
        assert self._start, "Timer %s has not been started yet." % self._name
        t = self._timer() - self._start
        self._timings.append(t)
        self._start = None
        return t

    def reset(self):
        """Reset the timer."""
        self._timings = []

    def add(self, t):
        """Add a timing."""
        if self._name not in Timer._timers:
            Timer._timers[self._name] = self
        self._timings.append(t)

    @property
    def name(self):
        """Name of the timer."""
        return self._name

    @property
    def elapsed(self):
        """Elapsed time for the currently running timer."""
        assert self._start, "Timer %s has not been started yet." % self._name
        return self._timer() - self._start

    @property
    def ncalls(self):
        """Total number of recorded events."""
        return len(self._timings)

    @property
    def total(self):
        """Total time spent for all recorded events."""
        return sum(self._timings)

    @property
    def average(self):
        """Average time spent per recorded event."""
        return np.average(self._timings)

    @property
    def c_time_total(self):
        """Total time spent for all recorded C timed events."""
        return self.ncalls * min(self._c_timings)

    @property
    def c_time_average(self):
        """Average time spent per recorded event."""
        return min(self._c_timings)

    @property
    def dv(self):
        if self._volume:
            return self.ncalls * self._volume / (1024.0 * 1024.0)
        return 0.0

    @property
    def bw(self):
        return self.dv / self.total

    @property
    def c_bw(self):
        return self._volume / (min(self._c_timings) * 1024.0 * 1024.0) if self.c_time_total else 0.0

    @property
    def c_mvbw(self):
        return self._volume_mvbw / (min(self._c_timings) * 1024.0 * 1024.0) if self.c_time_total else 0.0

    @property
    def c_mbw(self):
        return self._volume_mbw / (min(self._c_timings) * 1024.0 * 1024.0) if self.c_time_total else 0.0

    @property
    def c_rvbw(self):
        return self._volume / (max(self._c_rand_timings) * 1024.0 * 1024.0) if self.c_time_total else 0.0

    @property
    def sd(self):
        """Standard deviation of recorded event time."""
        return np.std(self._timings)

    @property
    def c_time_sd(self):
        """Standard deviation of recorded event C time."""
        return np.std(self._c_timings)

    @classmethod
    def summary(cls, filename=None):
        """Print a summary table for all timers or write CSV to filename."""
        if not cls._timers:
            return
        column_heads = ("Timer", "Total time", "Calls", "Average time", "Standard Deviation", "Tot C time", "Avg C time",
                        "Tot DV (MB)", "Tot BW (MB/s)", "Tot C BW (MB/s)")
        if isinstance(filename, str):
            import csv
            with open(filename, 'wb') as f:
                f.write(','.join(column_heads) + "\n")
                dialect = csv.excel
                dialect.lineterminator = '\n'
                w = csv.writer(f, dialect=dialect)
                w.writerows([(t.name, t.total, t.ncalls, t.average)
                            for t in cls._timers.values()])
        else:
            namecol = max([len(column_heads[0])] + [len(t.name)
                          for t in cls._timers.values()])
            totalcol = max([len(column_heads[1])] + [len('%g' % t.total)
                           for t in cls._timers.values()])
            ncallscol = max([len(column_heads[2])] + [len('%d' % t.ncalls)
                            for t in cls._timers.values()])
            averagecol = max([len(column_heads[3])] + [len('%g' % t.average)
                             for t in cls._timers.values()])
            sdcol = max([len(column_heads[4])] + [len('%g' % t.sd)
                        for t in cls._timers.values()])
            c_totalcol = max([len(column_heads[5])] + [len('%g' % t.c_time_total)
                             for t in cls._timers.values()])
            c_averagecol = max([len(column_heads[6])] + [len('%g' % t.c_time_average)
                               for t in cls._timers.values()])
            dvcol = max([len(column_heads[7])] + [len('%g' % t.dv)
                        for t in cls._timers.values()])
            bwcol = max([len(column_heads[8])] + [len('%g' % t.bw)
                        for t in cls._timers.values()])
            c_bwcol = max([len(column_heads[8])] + [len('%g' % t.c_bw)
                          for t in cls._timers.values()])
            c_mbwcol = max([len(column_heads[8])] + [len('%g' % t.c_mbw)
                           for t in cls._timers.values()])
            c_mvbwcol = max([len(column_heads[8])] + [len('%g' % t.c_mvbw)
                            for t in cls._timers.values()])
            c_rvbwcol = max([len(column_heads[8])] + [len('%g' % t.c_rvbw)
                            for t in cls._timers.values()])

            fmt = "%%%ds | %%%dg | %%%dd | %%%dg | %%%dg |  %%%dg | %%%dg | %%%dg | %%%dg | %%%dg | %%%dg | %%%dg | %%%dg | %%%dg" % (
                namecol, totalcol, ncallscol, averagecol, sdcol, c_totalcol, c_averagecol, dvcol, bwcol, 3, c_bwcol, c_mbwcol, c_mvbwcol, c_rvbwcol)
            keys = sorted(cls._timers.keys(), key=lambda k: k[-6:])
            if cls._output_file is not None:
                fmt += "\n"
            for k in keys:
                t = cls._timers[k]
                xtra_param = -1
                if cls._extra_param is not None:
                    xtra_param = cls._extra_param
                if cls._gflops > 0.0:
                    xtra_param = cls._flp_ops
                if cls._output_file is not None:
                    with open(cls._output_file, "a") as f:
                        tbw = fmt % (t.name, t.total, t.ncalls, t.average, t.sd, t.c_time_total, t.c_time_average, t.dv, t.bw, xtra_param, t.c_bw, t.c_mbw, t.c_mvbw, t.c_rvbw)
                        f.write(tbw)
                else:
                    print fmt % (t.name, t.total, t.ncalls, t.average, t.sd, t.c_time_total, t.c_time_average, t.dv, t.bw, xtra_param, t.c_bw, t.c_mbw, t.c_mvbw, t.c_rvbw)

    @classmethod
    def get_timers(cls):
        """Return a dict containing all Timers."""
        return cls._timers

    @classmethod
    def reset_all(cls):
        """Clear all timer information previously recorded."""
        cls._output_file = None
        cls._extra_param = None
        cls._only_kernel = False
        if not cls._timers:
            return
        cls._timers = {}
        cls._gflops = 0.0
        cls._flp_ops = 0

    @classmethod
    def output_file(cls, value):
        """Set the output file name."""
        cls._output_file = value

    @classmethod
    def extra_param(cls, value):
        """Used for printing extra information about the run."""
        cls._extra_param = value

    @classmethod
    def only_kernel(cls, value):
        """Only time the kernel execution and return the value otherwise
        time the whole wrapper."""
        cls._only_kernel = value

    @classmethod
    def set_max_bw(cls, value):
        """Include a distribution of values per element."""
        cls._max_bw = value


@contextmanager
def hpc_profiling(t, name):
    timer = Timer("%s-%s" % (t, name))
    timer.start()
    yield
    timer.stop()


def add_data_volume(t, name, vol, vol_mvbw, vol_mbw):
    if not configuration['randomize']:
        Timer("%s-%s" % (t, name)).data_volume(vol, vol_mvbw, vol_mbw)


def add_c_time(t, name, time):
    if not configuration['randomize']:
        Timer("%s-%s" % (t, name)).c_time(time)
    else:
        Timer("%s-%s" % (t, name)).c_rand_time(time)


def add_papi_gflops(t, name, flp_ops, gflops):
    if not configuration['randomize']:
        Timer("%s-%s" % (t, name)).papi_gflops(flp_ops, gflops)


def summary(filename=None):
    """Print a summary table for all timers or write CSV to filename."""
    Timer.summary(filename)


def get_timers(reset=False):
    """Return a dict containing all Timers."""
    ret = Timer.get_timers()
    if reset:
        Timer.reset_all()
    return ret


def reset():
    """Clear all timer information previously recorded."""
    Timer.reset_all()


def output_file(value):
    """Set an output file for the profiling summary."""
    Timer.output_file(value)


def extra_param(value):
    """Pass in information about the run to distinguish it from other runs if necessary."""
    if not configuration['randomize']:
        Timer.extra_param(value)


def only_kernel(value):
    """Only insert instrumentation code and timers around the kernel. By Default
    the instrumentation and timers are set at the beginning."""
    Timer.only_kernel(value)