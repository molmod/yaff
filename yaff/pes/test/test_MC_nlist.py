from yaff import System, ForceField, log
import numpy as np
import pkg_resources

def test_MC_nlist():

	def random_rotation(pos):
		com = np.average(pos, axis=0)
		pos -= com
		while True:
			V1 = np.random.rand(); V2 = np.random.rand(); S = V1**2 + V2**2;
			if S < 1:
				break;
		theta = np.array([2*np.pi*(2*V1*np.sqrt(1-S)-0.5), 2*np.pi*(2*V2*np.sqrt(1-S)-0.5), np.pi*((1-2*S)/2)])
		R_x = np.array([[1, 0, 0],[0, np.cos(theta[0]), -np.sin(theta[0])],[0, np.sin(theta[0]), np.cos(theta[0])]])
		R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],[0, 1, 0],[-np.sin(theta[1]), 0, np.cos(theta[1])]])
		R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]),0],[np.sin(theta[2]), np.cos(theta[2]),0],[0, 0, 1]])
		R = np.dot(R_z, np.dot( R_y, R_x ))
		pos_new = np.zeros((len(pos), len(pos[0])))
		for i, p in enumerate(pos):
			pos_new[i] = np.dot(R, np.array(p).T)
		return pos_new + com

	def get_adsorbate_pos(adsorbate, rvecs):
		pos = adsorbate.pos
		pos = random_rotation(pos)
		pos -= np.average(pos, axis=0)
		new_com = np.random.rand()*rvecs[0] + np.random.rand()*rvecs[1] + np.random.rand()*rvecs[2]
		return pos + new_com

	system = System.from_file(pkg_resources.resource_filename(__name__, '../../data/test/CAU_13.chk'))
	
	N_system = len(system.pos)
	adsorbate = System.from_file(pkg_resources.resource_filename(__name__, '../../data/test/xylene.chk'))

	# Add 4 adsorbates
	pos = system.pos
	ffatypes = np.append(system.ffatypes, adsorbate.ffatypes)
	bonds = system.bonds
	numbers = system.numbers
	ffatype_ids = system.ffatype_ids
	charges = system.charges
	masses = system.masses

	for i in range(4):
		pos = np.append(pos, get_adsorbate_pos(adsorbate,system.cell.rvecs), axis=0)
		bonds = np.append(bonds, adsorbate.bonds + N_system + len(adsorbate.pos) * i,axis=0)
		numbers = np.append(numbers, adsorbate.numbers, axis=0)
		ffatype_ids = np.append(ffatype_ids, adsorbate.ffatype_ids + max(system.ffatype_ids) + 1, axis=0)
		charges = np.append(charges, adsorbate.charges, axis=0)
		masses = np.append(masses, adsorbate.masses, axis=0)

	system = System(numbers, pos, ffatypes=ffatypes, ffatype_ids=ffatype_ids, bonds=bonds,\
					rvecs = system.cell.rvecs, charges=charges, masses=masses)
	ff_full_nlist = ForceField.generate(system, pkg_resources.resource_filename(__name__, '../../data/test/parameters_CAU-13_xylene.txt'))
	E_full = ff_full_nlist.compute()
	ff_no_frame_frame_nlist = ForceField.generate(system, \
					pkg_resources.resource_filename(__name__, '../../data/test/parameters_CAU-13_xylene.txt'),mc=True,n_frame=N_system)
	E_no_frame_frame = ff_no_frame_frame_nlist.compute()

	# Test 100 random configurations
	for i in range(100):
		new_pos = ff_full_nlist.system.pos
		for i in range(4):
			new_pos[N_system+i*len(adsorbate.pos):N_system+(i+1)*len(adsorbate.pos)] = get_adsorbate_pos(adsorbate,system.cell.rvecs)

		ff_full_nlist.update_pos(new_pos)
		ff_no_frame_frame_nlist.update_pos(new_pos)

		assert (ff_full_nlist.compute() - E_full) - (ff_no_frame_frame_nlist.compute() - E_no_frame_frame) < 10e-8







