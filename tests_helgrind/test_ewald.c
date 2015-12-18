#include <stdio.h>
#include "ewald.h"
#include "cell.h"
#include "constants.h"
#include <stdlib.h>

int main ()
{
  double L = 20.0;
  long natom = 3;
  long i;
  cell_type* cell = cell_new();
  double rvecs[9];
  rvecs[0] = L;
  rvecs[1] = 0.0;
  rvecs[2] = 0.0;
  rvecs[3] = 0.0;
  rvecs[4] = L;
  rvecs[5] = 0.0;
  rvecs[6] = 0.0;
  rvecs[7] = 0.0;
  rvecs[8] = L;
  double gvecs[9];
  gvecs[0] = M_TWO_PI/L;
  gvecs[1] = 0.0;
  gvecs[2] = 0.0;
  gvecs[3] = 0.0;
  gvecs[4] = M_TWO_PI/L;
  gvecs[5] = 0.0;
  gvecs[6] = 0.0;
  gvecs[7] = 0.0;
  gvecs[8] = M_TWO_PI/L;

  cell_update(cell, rvecs, gvecs, 3);

  double pos[3*natom];
  double charges[natom];
  for (i=0;i<3*natom;i++) {
    pos[i] = (float)rand()/(float)(RAND_MAX/L);
  }
  for (i=0;i<natom;i++) {
    charges[i] = (float)rand()/(float)(RAND_MAX/2.0) - 2.0;
  }
  double work[2*natom];
  long gmax[3];
  gmax[0] = 3;
  gmax[1] = 2;
  gmax[2] = 1;

  double E = compute_ewald_reci(pos,natom,charges,cell,1.0,gmax,2.0,20.0,NULL,work,NULL);
}
