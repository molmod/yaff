# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
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
#--


import numpy as np

from yaff.log import log


__all__ = ['random_opt', 'gauss_opt']


def random_opt(fn, x0, on_lower=None):
    def report(action, f, scale, threshold):
        if log.do_low:
           log('%10s: % 10.5e   %10.5e   %10.5e' % (action, f, scale, threshold))

    with log.section('RNDOPT'):
        scale = 0.01
        threshold = 0.0
        x = x0.copy()
        f = fn(x)
        report('Initial', f, scale, threshold)
        while scale > 1e-5:
            x1 = x + np.random.normal(0, scale, x.shape)
            f1 = fn(x1)
            if f1 < f + threshold:
                if f1 < f:
                    scale *= 1.1
                    threshold = f - f1
                    report('Lower', f1, scale, threshold)
                    if on_lower is not None:
                        on_lower(x1)
                else:
                    threshold *= 0.5
                    report('Higher', f1, scale, threshold)
                f = f1
                x = x1
            else:
                scale *= 0.999
                report('Useless', f1, scale, threshold)
        return x


class GaussianModel(object):
    def __init__(self, center, on_lower=None):
        self.center = center
        self.on_lower = on_lower

        self.npar = len(self.center)
        self.nrec_min = self.npar*(self.npar+1) + 3*self.npar
        self.nrec_max = 2*self.nrec_min

        self.history = []
        self.best = None
        self.counter = 0
        self.bad_counter = 0
        self.grad = np.zeros(self.npar, float)
        self.hess = np.identity(self.npar, float)
        self.temperature = 1e-2

    def sample(self):
        evals, evecs = np.linalg.eigh(self.hess)
        ridge = 1e-2*abs(evals).max()
        evals_inv = abs(evals)/(ridge**2 + evals**2)

        center = self.center - np.dot(evecs, np.dot(evecs.T, self.grad)*evals_inv)

        x1 = np.random.normal(0, self.temperature, self.npar)
        x1 = np.dot(evecs, np.dot(evecs.T, x1)*evals_inv) + center

        if log.do_medium:
            log('Sample   T = %10.5e   ridge = %10.5e   eval_min = %10.5e' % (self.temperature, ridge, abs(evals).min()))
        return x1

    def feed(self, x, f):
        self.counter += 1
        if log.do_low:
            log('Iteration %i' % self.counter)
        self.control_temperature(x, f)
        self.history.append((x, f))
        self.analyze()
        self.fit_model()
        if log.do_medium:
            log.hline()

    def control_temperature(self, x, f):
        if self.best is None or f > self.best[1]:
            if self.on_lower is not None:
                self.on_lower(x)
            self.bad_counter += 1
        else:
            self.bad_counter = 0
        if self.bad_counter > self.nrec_max:
            self.temperature *= 0.1
            self.bad_counter = 0
        if log.do_medium:
            log('Bad counter = %i/%i' % (self.bad_counter, self.nrec_max))

    def fit_model(self):
        if log.do_low:
            log('History has %i records. At least %i required for fit' % (len(self.history), self.nrec_min))
        if len(self.history) < self.nrec_min:
            return
        self.center = self.best[0]

        dm = []
        ev = []
        for x, f in self.history:
            row = [1.0]
            row.extend(x-self.center)
            for i0 in xrange(self.npar):
                for i1 in xrange(i0+1):
                    if i0 == i1:
                        factor = 0.5
                    else:
                        factor = 1.0
                    row.append(factor*(x[i0]-self.center[i0])*(x[i1]-self.center[i1]))
            dm.append(row)
            ev.append(f - self.best[1])

        dm = np.array(dm)
        ev = np.array(ev)
        ridge = ev.max()*0.01
        #if ridge > 0:
        #    ws = 1/(ridge+ev)**2
        #    dm *= ws.reshape(-1,1)
        #    ev *= ws
        scales = np.sqrt((dm**2).sum(axis=0))
        dm = dm/scales

        U, S, Vt = np.linalg.svd(dm, full_matrices=False)
        ridge = abs(S).max()*1e-6
        if ridge > 0:
            S_inv = abs(S)/(ridge**2+S**2)
            alpha = np.dot(Vt.T, np.dot(U.T, ev)*S_inv)
            coeffs = alpha/scales
            self.grad[:] = coeffs[1:self.npar+1]
            counter = self.npar+1
            for i0 in xrange(self.npar):
                for i1 in xrange(i0+1):
                    self.hess[i0,i1] = coeffs[counter]
                    self.hess[i1,i0] = coeffs[counter]
                    counter += 1

        if log.do_low:
            mv = np.dot(dm, alpha)
            rmsd = np.sqrt(((mv-ev)**2).mean())
            rms = np.sqrt(ev**2).mean()
            log('Fit RMSD = %10.5e    RMS = %10.5e    RE = %10.2f%%' % (rmsd, rms, rmsd/rms*100))
        if log.do_medium:
            log('Gradient:')
            log('  %s' % (' '.join('%7.1e' % g for g in self.grad)))
            evals = np.linalg.eigvalsh(self.hess)
            log('Hessian eigen values:')
            log('  %s' % (' '.join('%7.1e' % v for v in evals)))
            log('Function values in history:')
            fs = [f for x, f in self.history]
            fs.sort()
            log('  %s' % (' '.join('%7.1e' % f for f in fs)))


    def analyze(self):
        fs = np.array([f for x, f in self.history])
        self.best = self.history[fs.argmin()]
        if len(self.history) > self.nrec_max:
            if log.do_medium:
                log('Pruning worst point from history: f = % 10.5e' % fs.max())
            del self.history[fs.argmax()]


def gauss_opt(fn, x0, on_lower=None):
    with log.section('GAUOPT'):
        if log.do_low:
            log('Number of parameters: %i' % len(x0))
        gm = GaussianModel(x0, on_lower)
        x = x0.copy()
        f = fn(x)
        gm.feed(x, f)
        while gm.temperature > 1e-6:
            x1 = gm.sample()
            f1 = fn(x1)
            gm.feed(x1, f1)
        return gm.best[0]
