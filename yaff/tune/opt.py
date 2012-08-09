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
    def __init__(self, label, npar, minrec):
        self.label = label
        self.npar = npar
        self.minrec = minrec
        self.safeminrec = (self.minrec*3)/2

    def _fit_model(self, dm, ev):
        # Take only the best records
        dm = dm[:self.safeminrec]
        ev = ev[:self.safeminrec]

        # Find a stable solution
        scales = np.sqrt((dm**2).sum(axis=0))
        dm = dm/scales
        U, S, Vt = np.linalg.svd(dm, full_matrices=False)
        if abs(S).max() > 1e10*abs(S).min():
            if log.do_medium:
                log('  Ill-defined model')
                log(' '.join('%10.5e' % value for value in S))
            return False

        alpha = np.dot(Vt.T, np.dot(U.T, ev)/S)

        # check the rmsd
        mv = np.dot(dm, alpha)
        rmsd = np.sqrt(((mv-ev)**2).mean())
        rms = np.sqrt(ev**2).mean()
        if log.do_medium:
            log('Fit RMSD = %10.5e    RMS = %10.5e    RE = %10.2f%%' % (rmsd, rms, rmsd/rms*100))
        if rmsd>0.5*rms:
            if log.do_medium:
                log('  Poor fit')
            return False

        # return resulr
        return alpha/scales

    def rebuild(self, parhist, fnhist):
        valid = len(parhist) >= self.safeminrec
        if not valid and log.do_medium:
            log('  Too few records. At least %i are needed.' % self.safeminrec)
        return valid

    def sample(self):
        raise NotImplementedError


class FallbackModel(GaussianModel):
    def __init__(self, label, npar, sigma):
        self.sigma = sigma
        self.scaled_sigma = sigma
        GaussianModel.__init__(self, label, npar, 0)

    def rebuild(self, parhist, fnhist, scale):
        self.center = parhist[0]
        self.scaled_sigma = self.sigma*scale
        return True

    def sample(self):
        return self.center + np.random.normal(0, self.scaled_sigma, self.npar)


class DiagGaussianModel(GaussianModel):
    def __init__(self, label, npar, do_gradient):
        self.do_gradient = do_gradient
        minrec = npar + 1
        if do_gradient:
            minrec += npar
        GaussianModel.__init__(self, label, npar, minrec)

    def rebuild(self, parhist, fnhist, scale):
        valid = GaussianModel.rebuild(self, parhist, fnhist)
        if not valid:
            return False

        # Construct a set of equations to find the Hessian and optionally the
        # gradient
        self.center = parhist[0].copy()
        dm = []
        ev = []
        for x, f in zip(parhist, fnhist):
            row = [1.0]
            if self.do_gradient:
                row.extend(x-self.center)
            row.extend(0.5*(x-self.center)**2)
            dm.append(row)
            ev.append(f - fnhist[0])
        dm = np.array(dm)
        ev = np.array(ev)

        # solve the equations and determine of the result is valid.
        coeffs = self._fit_model(dm, ev)
        if coeffs is False:
            return False

        # extract gradient and Hessian diagonal from coeffs
        if self.do_gradient:
            gradient = coeffs[1:self.npar+1]
            counter = self.npar+1
        else:
            gradient = None
            counter = 1
        hdiag = coeffs[counter:counter+self.npar]

        # construct a Gaussian model that yields samples near the estimated
        # minimum with an expected function value below the mean of the current
        # set of samples
        tmp = hdiag.copy()
        if tmp.min() < 0:
            tmp -= tmp.min()
        threshold = 1e3
        if tmp.max() > threshold*tmp.min():
            tmp += (tmp.max() - threshold*tmp.min())/(threshold-1)
        hdiag_inv = 1/tmp

        parhist_sigmas = parhist.std(axis=0)
        self.sigmas = np.sqrt(hdiag_inv/hdiag_inv.max()*parhist_sigmas.max())*scale

        if self.do_gradient:
            # Compute the step
            step = -gradient*hdiag_inv*scale
            # Only apply the step if it is not ridiculously large.
            if (abs(step) < 10*self.sigmas).all():
                self.center += step

        if log.do_medium:
            log('Center:')
            log('  %s' % (' '.join('%7.1e' % v for v in self.center)))
            log('HDiag:')
            log('  %s' % (' '.join('%7.1e' % v for v in hdiag)))
            log('HDiag inv:')
            log('  %s' % (' '.join('%7.1e' % v for v in hdiag_inv)))
            log('Sigmas:')
            log('  %s' % (' '.join('%7.1e' % v for v in self.sigmas)))

        return True


    def sample(self):
        x = np.random.normal(0, 1, self.npar)*self.sigmas + self.center
        return x


class FullGaussianModel(GaussianModel):
    def __init__(self, label, npar, do_gradient):
        self.do_gradient = do_gradient
        minrec = (npar*(npar-1))/2 + 1
        if do_gradient:
            minrec += npar
        GaussianModel.__init__(self, label, npar, minrec)

    def rebuild(self, parhist, fnhist, scale):
        valid = GaussianModel.rebuild(self, parhist, fnhist)
        if not valid:
            return False

        # Construct a set of equations to find the Hessian and optionally the
        # gradient
        self.center = parhist[0].copy()
        dm = []
        ev = []
        for x, f in zip(parhist, fnhist):
            row = [1.0]
            if self.do_gradient:
                row.extend(x-self.center)
            for i0 in xrange(self.npar):
                for i1 in xrange(i0+1):
                    if i0 == i1:
                        factor = 0.5
                    else:
                        factor = 1.0
                    row.append(factor*(x[i0]-self.center[i0])*(x[i1]-self.center[i1]))
            dm.append(row)
            ev.append(f - fnhist[0])
        dm = np.array(dm)
        ev = np.array(ev)

        # solve the equations and determine of the result is valid.
        coeffs = self._fit_model(dm, ev)
        if coeffs is False:
            return False

        # extract gradient and Hessian from coeffs
        if self.do_gradient:
            gradient = coeffs[1:self.npar+1]
            counter = self.npar+1
        else:
            gradient = None
            counter = 1
        hessian = np.zeros((self.npar, self.npar), float)
        for i0 in xrange(self.npar):
            for i1 in xrange(i0+1):
                hessian[i0,i1] = coeffs[counter]
                hessian[i1,i0] = coeffs[counter]
                counter += 1

        # construct a Gaussian model that yields samples near the estimated
        # minimum with an expected function value below the mean of the current
        # set of samples
        evals, evecs = np.linalg.eigh(hessian)
        tmp = evals.copy()
        if tmp.min() < 0:
            tmp -= tmp.min()
        threshold = 1e3
        if tmp.max() > threshold*tmp.min():
            tmp += (tmp.max() - threshold*tmp.min())/(threshold-1)
        evals_inv = 1/tmp

        self.evecs = evecs
        parhist_t = np.dot(parhist[:self.safeminrec/2] - self.center, evecs)
        parhist_sigmas = parhist_t.std(axis=0)
        self.sigmas = np.sqrt(evals_inv/evals_inv.max()*parhist_sigmas.max())*scale

        if self.do_gradient:
            # Compute the step
            step = np.dot(evecs, np.dot(evecs.T, -gradient)*evals_inv)*scale
            # Only apply the step if it is not ridiculously large.
            if (abs(step) < 10*self.sigmas).all():
                self.center += step

        if log.do_medium:
            log('Center:')
            log('  %s' % (' '.join('%7.1e' % v for v in self.center)))
            log('Evals:')
            log('  %s' % (' '.join('%7.1e' % v for v in evals)))
            log('Evals inv:')
            log('  %s' % (' '.join('%7.1e' % v for v in evals_inv)))
            log('Sigmas:')
            log('  %s' % (' '.join('%7.1e' % v for v in self.sigmas)))

        return True


    def sample(self):
        x = np.random.normal(0, 1, self.npar)
        x = np.dot(self.evecs, np.dot(self.evecs.T, x)*self.sigmas) + self.center
        return x


class ParameterHistory(object):
    def __init__(self, npar, sigma, on_lower=None):
        self.npar = npar
        self.sigma = sigma
        self.on_lower = on_lower

        self.history = []
        self.counter = 0
        self.models = [
            FullGaussianModel('FGG', npar, True),
            FullGaussianModel('FG', npar, False),
            DiagGaussianModel('DGG', npar, True),
            DiagGaussianModel('DG', npar, False),
            FallbackModel('FB', npar, sigma),
        ]
        self.scale = 1.0
        self.maxrec = max(model.safeminrec for model in self.models)

    def sample(self):
        for model in self.models:
            if log.do_medium:
                log('Trying model %s' % model.label)
            if model.rebuild(self._parhist, self._fnhist, self.scale):
                x = model.sample()
                return x

    def feed(self, f, x):
        self.counter += 1
        if log.do_low:
            log.hline()
            log('Iteration %i' % self.counter)
        self.add_record(f, x)

        if log.do_low:
            log('Best   f    = %10.5e' % self.history[0][0])
            log('Median f    = %10.5e' % self.history[len(self.history)/2][0])
            log('Worst  f    = %10.5e' % self.history[-1][0])
            log('Newest f    = %10.5e' % f)
            log('# records   = %10i' % (len(self.history)))

        if len(self.history) > 1:
            # Change the size of the steps made by the last model if needed.
            # The size of the steps is defined with respect to the spread on the
            # current history of N best parameters times the scale factor
            # modified below.
            if f >= self.history[-1][0]:
                # Way to large function value compared to current results
                self.scale *= 0.5
            elif f < self.history[len(self.history)/2][0]:
                # Rather small function value compared to current results
                self.scale *= 1.01
            else:
                self.scale *= 0.9
            if log.do_medium:
                log('Scaling sigmas by %.1e' % self.scale)


    def add_record(self, f, x):
        if len(self.history) > 0:
            # Check if this one is the best so far
            if f < self.history[0][0]:
                if self.on_lower is not None:
                    self.on_lower(x)

        # Add the record
        self.history.append((f, x))
        self.history.sort()

        if len(self.history) > self.maxrec:
            del self.history[-1]

        # Transform to useful matrices
        self._fnhist = np.array([row[0] for row in self.history])
        self._parhist = np.array([row[1] for row in self.history])

    def converged(self, sigma_threshold):
        return len(self.history) == self.maxrec and (self._parhist.std(axis=0).max() <= sigma_threshold).all()
        return self._parhist.std(axis=0).max()

    def get_best_pars(self):
        return self._parhist[0]


def gauss_opt(fn, x0, sigma, sigma_threshold=1e-8, on_lower=None):
    with log.section('GAUOPT'):
        if log.do_low:
            log('Number of parameters: %i' % len(x0))
        ph = ParameterHistory(len(x0), sigma, on_lower)
        x = x0.copy()
        f = fn(x)
        ph.feed(f, x)
        while not ph.converged(sigma_threshold):
            x1 = ph.sample()
            f1 = fn(x1)
            ph.feed(f1, x1)
        return ph.get_best_pars()
