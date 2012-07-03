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


__all__ = ['random_opt', 'gauss_opt', 'rosenbrock_opt']


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


class Newton(object):
    def __init__(self, grad, hess):
        self.grad = grad
        self.hess = hess
        self.evals, self.evecs = np.linalg.eigh(self.hess)

    def __call__(self, ridge):
        self.evals_inv = 1/abs(ridge + self.evals)
        self.step = -np.dot(self.evecs, np.dot(self.evecs.T, self.grad)*self.evals_inv)
        return np.linalg.norm(self.step)


class GaussianModel(object):
    def __init__(self, center, sigma, on_lower=None):
        self.center = center
        self.on_lower = on_lower
        self.npar = len(self.center)
        self.nrec = self.npar*(self.npar+1) + 3*self.npar

        self.sigmas = np.ones(self.npar)*sigma
        self.evecs = np.identity(self.npar)
        #self.trust_radius = sigma
        self.temperature = None

        self.history = []
        self.counter = 0

    full = property(lambda self: len(self.history) >= self.nrec)

    overfull = property(lambda self: len(self.history) > self.nrec*2)

    def sample(self):
        x1 = np.random.normal(0, 1, self.npar)
        x1 = np.dot(self.evecs, np.dot(self.evecs.T, x1)*self.sigmas) + self.center
        if log.do_medium:
            log('Sample:')
            log('  %s' % (' '.join('%10.5e' % v for v in x1)))
        return x1

    def feed(self, f, x):
        self.counter += 1
        if log.do_low:
            log('Iteration %i' % self.counter)
            log('History has %i records. %i required for fit.' % (len(self.history), self.nrec))
        self.add_record(f, x)
        self.center = self.history[0][1].copy()
        if self.full:
            self.fit_model()
            #self.trust_radius *= 1.01

        if log.do_low:
            log('Best   f    = %10.5e' % self.history[0][0])
            log('Newest f    = %10.5e' % f)
            if self.temperature is not None:
                log('Temperature = %10.5e' % self.temperature)
        if log.do_medium:
            log('Center:')
            log('  %s' % (' '.join('%7.1e' % v for v in self.center)))
            log('History:')
            log('  %s' % (' '.join('%7.1e' % f for f, x in self.history)))
            log.hline()

    def add_record(self, f, x):
        if len(self.history) > 0:
            # Check if this one is the best so far
            if f < self.history[0][0]:
                if self.on_lower is not None:
                    self.on_lower(x)
        if self.temperature is not None:
            if f < self.history[0][0]:
                self.temperature *= 2.0
            elif f > self.history[-1][0]:
                self.temperature *= 0.5
            else:
                self.temperature *= 0.999

        # Add the record
        self.history.append((f, x))
        self.history.sort()

        if self.overfull:
            del self.history[-1]


    def fit_model(self):
        # Build a model for the hessian and the gradient

        dm = []
        ev = []
        for f, x in self.history:
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
            ev.append(f - self.history[0][0])

        dm = np.array(dm)
        ev = np.array(ev)
        #ridge = ev.mean()*0.01
        #if ridge > 0:
        #    ws = 1/(ridge+ev)**2
        #    dm *= ws.reshape(-1,1)
        #    ev *= ws
        scales = np.sqrt((dm**2).sum(axis=0))
        dm = dm/scales

        U, S, Vt = np.linalg.svd(dm, full_matrices=False)
        ridge = abs(S).max()*1e-3
        S_inv = abs(S)/(ridge**2+S**2)
        alpha = np.dot(Vt.T, np.dot(U.T, ev)*S_inv)
        coeffs = alpha/scales
        grad = coeffs[1:self.npar+1]
        counter = self.npar+1
        hess = np.zeros((self.npar, self.npar), float)
        for i0 in xrange(self.npar):
            for i1 in xrange(i0+1):
                hess[i0,i1] = coeffs[counter]
                hess[i1,i0] = coeffs[counter]
                counter += 1

        if log.do_medium:
            mv = np.dot(dm, alpha)
            rmsd = np.sqrt(((mv-ev)**2).mean())
            rms = np.sqrt(ev**2).mean()
            log('Fit RMSD = %10.5e    RMS = %10.5e    RE = %10.2f%%' % (rmsd, rms, rmsd/rms*100))

        evals, evecs = np.linalg.eigh(hess)
        threshold = evals.max()*0.1
        tmp = evals.copy()
        tmp[tmp<threshold] = threshold
        evals_inv = 1/tmp

        self.evecs = evecs
        if self.temperature is None:
            sigmas = np.sqrt(evals_inv)/ev[len(ev)/2]
            self.temperature = (self.sigmas/sigmas).max()

        self.sigmas = np.sqrt(evals_inv)/ev[len(ev)/2]*self.temperature
        step = np.dot(evecs, np.dot(evecs.T, -grad)*evals_inv)
        step *= np.linalg.norm(self.sigmas)/np.linalg.norm(step)
        self.center += step


        if log.do_medium:
            log('Evals:')
            log('  %s' % (' '.join('%7.1e' % v for v in evals)))
            log('Evals inv:')
            log('  %s' % (' '.join('%7.1e' % v for v in evals_inv)))
            log('Sigmas:')
            log('  %s' % (' '.join('%7.1e' % v for v in self.sigmas)))


def gauss_opt(fn, x0, sigma, on_lower=None):
    with log.section('GAUOPT'):
        if log.do_low:
            log('Number of parameters: %i' % len(x0))
        gm = GaussianModel(x0, sigma, on_lower)
        x = x0.copy()
        f = fn(x)
        gm.feed(f, x)
        while gm.sigmas.max() > 1e-8:
            x1 = gm.sample()
            f1 = fn(x1)
            gm.feed(f1, x1)
        return gm.history[0][1]


def rosenbrock_opt(fn, x0, small, xtol, on_lower=None):
    def my_fn(x):
        result = fn(x)
        if log.do_medium:
            log('    f = %10.5e' % result)
        return result

    def my_lower(f, x):
        if on_lower is not None:
            on_lower(x)
        if log.do_low:
            log('    **LOWER** f = %10.5e **LOWER** ' % f)
            # double check
            f = fn(x)
            log('    **CHECK** f = %10.5e **CHECK** ' % f)

    with log.section('ROSOPT'):

        f0 = my_fn(x0)
        npar = len(x0)
        basis = np.identity(npar, float)
        counter = 0
        while True:
            counter += 1
            if log.do_medium:
                log('iteration %i' % counter)
            x0_start = x0.copy()
            nostep = True

            # Try to make moves along the directions of basis
            for i in xrange(npar):
                if log.do_medium:
                    log('  decrease %i' % i)
                # try by decreasing
                delta = small
                nattempt = 0
                while abs(delta) > xtol:
                    x1 = x0 + delta*basis[i]
                    f1 = my_fn(x1)
                    if f1 < f0:
                        f0, x0 = f1, x1
                        my_lower(f0, x0)
                        nostep = False
                        break
                    if delta > 0:
                        delta *= -1
                    else:
                        delta *= -0.5
                    nattempt += 1

                # if no decrease was needed, try increasing
                if nattempt < 2:
                    if log.do_medium:
                        log('  increase %i' % i)
                    while True:
                        x1 = x0 + delta*basis[i]
                        f1 = my_fn(x1)
                        if f1 >= f0:
                            break
                        f0, x0 = f1, x1
                        my_lower(f0, x0)
                        delta *= 2
                        nostep = False

                # tried enough, new direction in next iteration

            # if the shortest possible step in each direction was tested, bail out
            if nostep:
                if log.do_medium:
                    log('Not a single successful step in this iteration. Giving up.')
                break

            # compute step.
            step = x0 - x0_start
            if log.do_medium:
                log('step = [%s]' % ' '.join('%10.5e' % v for v in step))

            # project each basis vector on ortho complement
            for i in xrange(npar):
                basis[i] -= step*np.dot(step, basis[i])/np.linalg.norm(step)**2

            # use svd to get the basis for the remainder
            U, S, Vt = np.linalg.svd(basis)
            basis[0] = step/np.linalg.norm(step)
            for i in xrange(1, npar):
                basis[i] = Vt[i-1]
            for i in xrange(npar):
                log('basis[%i] = [%s]' % (i, ' '.join('%10.5e' % v for v in basis[i])))

            small = np.linalg.norm(step)*0.01


        return x0
