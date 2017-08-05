#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
"""This example is a demonstration of the back-propagation approach discussed
   in the chapter "The back-propagation algorithm for the computation of energy
   derivatives".

   This file implements the Bead class discussed in that chapter and some
   derived classes. The test routines verify the analytical derivatives of
   the individual bead classes. This example does not depend of Yaff, but the
   tests do depend on the molmod module.
"""


from __future__ import print_function

import numpy as np

from molmod import check_delta


class Bead(object):
    # The name `Bead' stresses that each class implements a part of
    # a complete function to which the `Chain'-rule is applied.
    def __init__(self, nins, nout):
        self.nins = nins # list of sizes of the input arrays.
        self.nout = nout # size of the output array
        self.ar_out = np.zeros(nout) # the output of the function
        self.ar_gout = np.zeros(nout) # the derivative of the final scalar
                                      # function towards the outputs

    def forward(self, ars_in):
        '''Subclasses implement a mathematical function here.

           **Arguments:**

           ars_in
                A list of input arrays
        '''
        assert len(self.nins) == len(ars_in)
        for i in range(len(self.nins)):
            assert len(ars_in[i]) == self.nins[i]

    def back(self, ars_gin):
        '''Subclasses implement the chain rule for the mathematical function here.

           **Arguments:**

           ars_gin
                A list of output arrays for the derivatives of the final
                scalar towards the corresponding input arrays of the forward
                method. Results must be added to the arrays ars_gin, not
                overwritten.

           This routine assumes that the contents of self.ar_gout is already
           computed before this routine is called. The code in the subclass
           must transform 'the derivatives of the energy towards the output
           of this function' (self.ar_gout) into 'the derivatives of the
           energy towards the input of this function' (ars_gin).
        '''
        assert len(self.nins) == len(ars_gin)
        for i in range(len(self.nins)):
            assert len(ars_gin[i]) == self.nins[i]

    def resetg(self):
        # clear the gout array
        self.ar_gout[:] = 0


class BeadLinTransConst(Bead):
    def __init__(self, coeffs, consts):
        assert len(coeffs.shape) == 2 # must be a transformation matrix
        assert len(consts.shape) == 1 # must be a vector with constants
        assert consts.shape[0] == coeffs.shape[0]
        self.coeffs = coeffs
        self.consts = consts
        Bead.__init__(self, [coeffs.shape[1]], coeffs.shape[0])

    def forward(self, ars_in):
        Bead.forward(self, ars_in)
        self.ar_out[:] = np.dot(self.coeffs, ars_in[0]) + self.consts

    def back(self, ars_gin):
        Bead.back(self, ars_gin)
        ars_gin[0][:] += np.dot(self.coeffs.T, self.ar_gout)


class BeadSwitch(Bead):
    def __init__(self, size):
        Bead.__init__(self, [size], size)

    def forward(self, ars_in):
        Bead.forward(self, ars_in)
        self.ar_out[:] = np.tanh(ars_in[0])

    def back(self, ars_gin):
        Bead.back(self, ars_gin)
        # Recycle the intermediate result from the forward computation...
        # This happens quite often in real-life code.
        ars_gin[0][:] += self.ar_gout*(1-self.ar_out**2)


class BeadDot(Bead):
    def __init__(self, nin):
        Bead.__init__(self, [nin, nin], 1)

    def forward(self, ars_in):
        Bead.forward(self, ars_in)
        self.ar_out[0] = np.dot(ars_in[0], ars_in[1])
        # keep a hidden reference to the input arrays
        self._ars_in = ars_in

    def back(self, ars_gin):
        Bead.back(self, ars_gin)
        ars_gin[0][:] += self.ar_gout[0]*self._ars_in[1]
        ars_gin[1][:] += self.ar_gout[0]*self._ars_in[0]



def check_bead_delta(bead, amp, eps):
    '''General derivative testing routine for the Bead classes

       **Arguments:**

       bead
            An instance of a subclass of the Bead class.

       amp
            Amplitude for the randomly generated reference input data for the
            bead.

       eps
            Magnitude of the small displacements around the reference input
            data.
    '''
    # A wrapper around the bead that matches the API of the molmod derivative
    # tester.
    def fun(x, do_gradient=False):
        # chop the contiguous array x into ars_in
        ars_in = []
        offset = 0
        for nin in bead.nins:
            ars_in.append(x[offset:offset+nin])
            offset += nin
        # call forward path
        bead.forward(ars_in)
        # to gradient or not to gradient ...
        if do_gradient:
            # call back path for every output component
            gxs = []
            for i in range(bead.nout):
                bead.resetg()
                bead.ar_gout[i] = 1
                ars_gin = [np.zeros(nin) for nin in bead.nins]
                bead.back(ars_gin)
                gx = np.concatenate(ars_gin)
                gxs.append(gx)
            gxs = np.array(gx)
            return bead.ar_out, gxs
        else:
            return bead.ar_out

    nx = sum(bead.nins)
    x = np.random.uniform(-amp, amp, nx)
    dxs = np.random.uniform(-eps, eps, (100, nx))
    check_delta(fun, x, dxs)


def test_bead_lin_trans_const():
    coeffs = np.random.normal(0, 1, (8, 5))
    consts = np.random.normal(0, 1, 8)
    bead = BeadLinTransConst(coeffs, consts)
    check_bead_delta(bead, 1.0, 1e-4)

def test_bead_switch():
    bead = BeadSwitch(8)
    check_bead_delta(bead, 1.0, 1e-4)

def test_bead_dot():
    bead = BeadDot(5)
    check_bead_delta(bead, 1.0, 1e-4)


def test_neural_net():
    # A simple neural network implementation based on the Bead classes
    # This example implements a three-layer (5,4,1) network with random
    # parameters.
    ltc1 = BeadLinTransConst(np.random.normal(0,1,(4,5)), np.random.normal(0,1,4))
    swi1 = BeadSwitch(4)
    ltc2 = BeadLinTransConst(np.random.normal(0,1,(1,4)), np.random.normal(0,1,1))
    swi2 = BeadSwitch(1)

    # This neural net routine does not need to know the sizes of each layer.
    # Exercise: generalize it such that it would work for an arbitrary number of
    #           layers.
    def neural_net(x, do_gradient=False):
        # forward path
        ltc1.forward([x])
        swi1.forward([ltc1.ar_out])
        ltc2.forward([swi1.ar_out])
        swi2.forward([ltc2.ar_out])
        # back path
        if do_gradient:
            # clean gradient arrays
            ltc1.resetg()
            swi1.resetg()
            ltc2.resetg()
            swi2.resetg()
            # compute derivatives
            gx = np.zeros(x.shape)
            swi2.ar_gout[0] = 1.0 # we know the the final output is a scalar.
            swi2.back([ltc2.ar_gout])
            ltc2.back([swi1.ar_gout])
            swi1.back([ltc1.ar_gout])
            ltc1.back([gx])
            return swi2.ar_out[0], gx # we know the the final output is a scalar.
        else:
            return swi2.ar_out[0] # we know the the final output is a scalar.

    x = np.random.normal(0,1,5)
    print('The inputs for the neural network')
    print(x)
    print()

    print('Calling neural network without gradient')
    print('F(x)', neural_net(x))
    print()

    print('Calling neural network with gradient')
    f, gx = neural_net(x, True)
    print('F(x)', f)
    print('Gradient')
    print(gx)
    print()

    print('Running check_delta on the neural network function.')
    dxs = np.random.normal(0,1e-4,(100,5))
    check_delta(neural_net, x, dxs)
    print('Test passed.')

if __name__ == '__main__':
    test_bead_lin_trans_const()
    test_bead_switch()
    test_bead_dot()
    test_neural_net()
