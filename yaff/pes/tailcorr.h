// YAFF is yet another force-field code.
// Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
// --
double pair_tailcorr_cut_lj(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_lj(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_mm3(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_mm3(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_grimme(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_grimme(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_exprep(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_exprep(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_qmdffrep(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_qmdffrep(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_ljcross(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_ljcross(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_dampdisp(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_dampdisp(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_disp68bjdamp(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_disp68bjdamp(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_ei(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_ei(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_eidip(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_eidip(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_eislater1s1scorr(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_eislater1s1scorr(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_eislater1sp1spcorr(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_eislater1sp1spcorr(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_olpslater1s1s(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_olpslater1s1s(void *pair_data, long center_index, long other_index, double rcut, double width);
double pair_tailcorr_cut_chargetransferslater1s1s(void *pair_data, long center_index, long other_index, double rcut);
double pair_tailcorr_switch3_chargetransferslater1s1s(void *pair_data, long center_index, long other_index, double rcut, double width);
