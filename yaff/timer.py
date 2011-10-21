# -*- coding:utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import time
from contextlib import contextmanager


__all__ = ['timer']


class Timer(object):
    def __init__(self):
        self.cpu = 0.0
        self._start = None

    def start(self):
        assert self._start is None
        self._start = time.clock()

    def stop(self):
        assert self._start is not None
        self.cpu += time.clock() - self._start
        self._start = None


class SubTimer(object):
    def __init__(self):
        self.total = Timer()
        self.own = Timer()

    def start(self):
        self.total.start()
        self.own.start()

    def start_sub(self):
        self.own.stop()

    def stop_sub(self):
        self.own.start()

    def stop(self):
        self.own.stop()
        self.total.stop()


class TimerGroup(object):
    def __init__(self):
        self.parts = {}
        self.stack = []

    def reset(self):
        for timer in self.parts.itervalues():
            timer.total.cpu = 0.0
            timer.own.cpu = 0.0

    @contextmanager
    def section(self, label):
        self._start(label)
        try:
            yield
        finally:
            self._stop()

    def _start(self, label):
        # get the right timer object
        timer = self.parts.get(label)
        if timer is None:
            timer = SubTimer()
            self.parts[label] = timer
        # start timing
        timer.start()
        if len(self.stack) > 0:
            self.stack[-1].start_sub()
        # put it on the stack
        self.stack.append(timer)

    def _stop(self):
        timer = self.stack.pop(-1)
        timer.stop()
        if len(self.stack) > 0:
            self.stack[-1].stop_sub()

    def get_max_own_cpu(self):
        result = None
        for part in self.parts.itervalues():
            if result is None or result < part.own.cpu:
                result = part.own.cpu
        return result

    def report(self, log):
        max_own_cpu = self.get_max_own_cpu()
        #if max_own_cpu == 0.0:
        #    return
        with log.section('TIMER'):
            log('Overview of CPU time usage.')
            log.hline()
            log('Label          Total   Own')
            log.hline()
            bar_width = log.width-27
            for label, timer in sorted(self.parts.iteritems()):
                #if timer.total.cpu == 0.0:
                #    continue
                if max_own_cpu > 0:
                    cpu_bar = "W"*int(timer.own.cpu/max_own_cpu*bar_width)
                else:
                    cpu_bar = ""
                log('%14s %5.1f %5.1f %30s' % (
                    label.ljust(14),
                    timer.total.cpu, timer.own.cpu, cpu_bar.ljust(bar_width),
                ))
            log.hline()


timer = TimerGroup()
