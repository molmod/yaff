// YAFF is yet another force-field code
// Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
// Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
// (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
// stated.
//
// This file is part of YAFF.
//
// YAFF is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// YAFF is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>
//
//--

#include <stdlib.h>
#include <math.h>
#include "constants.h"

double slaterei_0_0(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g){
  /* Radial part of the electrostatic interaction between two sites separated
     by a distance d.
     The first site contains
        + a Slater monopole Na
        + a point monopole Za
     The second site contains
        + a Slater monopole Nb
        + a point monopole Zb
     The long-range part of the interaction is NOT included; it is given by
        (Na+Za)*(Nb+Zb)/d
     and has to be computed elsewhere. The reason for this separation of the
     long range part is so that conventional techniques (Ewald summation, Wolff
     summation) can be applied without alteration.
  */
  double pot1, pot2, pot3, pot_tmp;
  double g1, g2, g3;
  g3 = 0.0;
  // Precompute some powers and other factors
  double da = d/a;
  double db = d/b;
  double expa = exp(-da);
  double expb = exp(-db);
  double a2 = a*a;
  double a3 = a2*a;
  double a4 = a2*a2;
  double b2 = b*b;
  double b3 = b2*b;
  double b4 = b2*b2;
  // Point-Slater [expa]
  pot1 = -(1.0+0.5*da)*expa/d;
  if (g != NULL) g1 = -(1.0/a+1.0/d)*pot1 - 0.5*expa/d/a;
  // Point-Slater [expb]
  pot2 = -(1.0+0.5*db)*expb/d;
  if (g != NULL) g2 = -(1.0/b+1.0/d)*pot2 - 0.5*expb/d/b;
  // Discriminate between small and large difference in Slater width
  if (fabs(a-b) > 0.025) {
    // Precompute some more factors
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    // Slater-Slater [expa]
    pot3  = -( (a2-3.0*b2)*diff3 + 0.5*diff2*da )*a4*expa/d;
    if (g != NULL) g3 = -(1.0/a+1.0/d)*pot3 - 0.5*diff2*a3*expa/d;
    // Slater-Slater [expb]
    pot_tmp = -( (3.0*a2-b2)*diff3 + 0.5*diff2*db )*b4*expb/d;
    if (g != NULL) g3 += -(1.0/b+1.0/d)*pot_tmp - 0.5*diff2*b3*expb/d;
    pot3 += pot_tmp;
  } else {
    // Precompute some more factors
    double da2 = da*da;
    double da3 = da2*da;
    double da4 = da2*da2;
    double delta = a-b;
    double delta2 = delta*delta;
    // 0-th order Taylor [expa]
    pot3  = -(48.0+33.0*da+9.0*da2+da3)/48.0/d*expa;
    if (g != NULL) g3 = -(1.0/d+1.0/a)*pot3 - (33.0+18.0*da+3.0*da2)*expa/48.0/d/a;
    // 1-st order Taylor [expa]
    pot_tmp = (15.0+15.0*da+6.0*da2+da3)/96.0/a2*expa*delta;
    if (g != NULL) g3 += -1.0/a*pot_tmp + (15.0+12.0*da+3.0*da2)/96.0/a3*expa*delta;
    pot3 += pot_tmp;
    // 2-nd order Taylor [expa]
    pot_tmp = (60.0+60.0*da+15.0*da2-5.0*da3-3.0*da4)/480.0/a3*expa*delta2/2.0;
    if (g != NULL) g3 += -1.0/a*pot_tmp + (60.0+30.0*da-15.0*da2-12.0*da3)/480.0/a4*expa*delta2/2.0;
    pot3 += pot_tmp;
  }
  double pot = Na*Zb*pot1 + Za*Nb*pot2 + Na*Nb*pot3;
  if (g != NULL) *g = (Na*Zb*g1 + Za*Nb*g2 + Na*Nb*g3)/d;
  return pot;
}
/*

double ei_1_0(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g){
    double da = d/a;
    double db = d/b;
    double a2 = a*a;
    double a3 = a2*a;
    double a4 = a2*a2;
    double b2 = b*b;
    double b3 = b2*b;
    double b4 = b2*b2;
    double b5 = b3*b2;
    double b6 = b3*b3;
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    double diff4 = diff2*diff2;

    // r^-3
    double a_0 = a4*(a4-4.0*a2*b2+6.0*b4)*diff4;
    double a_1 = a3*(a4-4.0*a2*b2+6.0*b4)*diff4;
    double a_2 = 0.5*a2*(a2-3.0*b2)*diff3;
    double a_3 = 0.125*a*diff2;
    // (1.0+db+0.5*db2+0.125*db3)/d3
    double b_0 = b6*(4.0*a2-b2)*diff4;
    double b_1 = b5*(4.0*a2-b2)*diff4;
    double b_2 = 0.5*b4*diff3;
    // (1.0+db+0.5*db*db)/d3
    return 0.0;
}

double ei_1_1(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g){
    double da = d/a;
    double db = d/b;
    double a2 = a*a;
    double a3 = a2*a;
    double a4 = a2*a2;
    double a5 = a3*a2;
    double a6 = a3*a3;
    double b2 = b*b;
    double b3 = b2*b;
    double b4 = b2*b2;
    double b5 = b3*b2;
    double b6 = b3*b3;
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    double diff4 = diff2*diff2;
    double diff5 = diff3*diff2;

    // r^-5
    double a_0 = 3.0*a6*(a4-5.0*a2*b2+10.0*b4)*diff5;
    double a_1 = 3.0*a5*(a4-5.0*a2*b2+10.0*b4)*diff5;
    double a_2 = 1.5*a4*(a4-5.0*a2*b2+8.0*b4)*diff5;
    double a_3 = 0.5*a3*(a2-4.0*b2)*diff4;
    double a_4 = 0.125*a2*diff3;

    return 0.0;
}

double ei_1_1_kronecker(double a, double b, double Na, double Za, double Nb, double Zb, double d, double *g){
    double da = d/a;
    double db = d/b;
    double a2 = a*a;
    double a3 = a2*a;
    double a4 = a2*a2;
    double a5 = a3*a2;
    double a6 = a3*a3;
    double b2 = b*b;
    double b3 = b2*b;
    double b4 = b2*b2;
    double b5 = b3*b2;
    double b6 = b3*b3;
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    double diff4 = diff2*diff2;
    double diff5 = diff3*diff2;
    // r^-3
    double a_0 = 1.0*a6*(a4-5.0*a2*b2+10.0*b4)*diff5;
    double a_1 = 1.0*a5*(a4-5.0*a2*b2+10.0*b4)*diff5;
    double a_2 = 0.5*a4*(a2-4.0*b2)*diff4;
    double a_3 = 0.125*a3*diff3;
    return 0.0;
}
*/




double slaterolp_0_0(double a, double b, double d, double *g){
  /* Radial part of the overlap between two sites separated
     by a distance d. Both sites contain a unit Slater monopole.
     There is a different interface compared to the electrostatic interaction
     code because:
       + Possible point monopoles do not contribute to the overlap.
       + There is no long range term present in the overlap expression.
     TODO: clean up code below.
  */
  double delta, da, db;
  double pot = 0.0;
  delta = a - b;
  da = d/a;
  db = d/b;
  // Discriminate between small and not small difference in slater width
  if (fabs(delta) > 0.025) {
    double a2 = a*a;
    double b2 = b*b;
    double diff = 1.0/(a2-b2);
    double diff2 = diff*diff;
    double diff3 = diff2*diff;
    double pot1 = 0.5*(-4.0*a2*b2*diff3 + a*d*diff2)*exp(-da)/d/M_FOUR_PI;
    double pot2 = 0.5*( 4.0*a2*b2*diff3 + b*d*diff2)*exp(-db)/d/M_FOUR_PI;
    pot += pot1 + pot2;
    if (g != NULL) {
      *g = -pot/d/d-pot1/d/a-pot2/d/b + 0.5*a*diff2*exp(-da)/d/M_FOUR_PI/d + 0.5*b*diff2*exp(-db)/d/M_FOUR_PI/d;
    }
  } else {
    double da2 = da*da;
    double da3 = da2*da;
    double da4 = da2*da2;
    double a2i = 1.0/(a*a);
    double a3i = a2i/a;
    double a4i = a2i*a2i;
    double a5i = a3i*a2i;
    double a6i = a3i*a3i;
    pot += (da2+3.0*da+3.0)*exp(-da)*a3i/48.0/M_FOUR_PI;
    pot += (-da3+2.0*da2+9.0*da+9.0)*exp(-da)*a4i/96.0/M_FOUR_PI*delta;
    pot += (3.0*da4-25.0*da3+5.0*da2+90.0*da+90.0)*exp(-da)*a5i/960.0/M_FOUR_PI*delta*delta;
    if (g != NULL) {
        *g  = -pot/d/a;
        *g += (3.0+2.0*da)*exp(-da)*a4i/48.0/M_FOUR_PI/d;
        *g += (9.0+4.0*da-3.0*da2)*exp(-da)*a5i/96.0/M_FOUR_PI*delta/d;
        *g += (90.0+10.0*da-75.0*da2+12.0*da3)*exp(-da)*a6i/960.0/M_FOUR_PI*delta*delta;
    }
  }
  return pot;
}
