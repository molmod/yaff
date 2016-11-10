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


cimport numpy as np
cimport nlist
cimport truncation

cdef extern from "slater.h":
    double slaterei_0_0(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g)
    double slaterei_1_0(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g)
    double slaterei_1_1(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g)
    double slaterei_1_1_kronecker(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g)
    double slaterolp_0_0(double a, double b, double d, double *g)
