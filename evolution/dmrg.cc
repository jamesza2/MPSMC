#include "itensor/all.h"
#include "../headers/input.h"
#include "../headers/operators.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>

void apply_MPO_no_truncation(itensor::MPO &itev, itensor::MPS &psi){
	int num_sites = itensor::length(psi);

	std::vector<itensor::Index> new_link_indices;
	new_link_indices.reserve(num_sites);

	auto MPO_first_link = itensor::rightLinkIndex(itev, 1);
	auto MPS_first_link = itensor::rightLinkIndex(psi, 1);
	auto [first_combiner, first_link_index] = itensor::combiner(itensor::IndexSet(MPO_first_link, MPS_first_link),{"Tags=","Link,l=1"});

	std::vector<itensor::ITensor> new_MPS;
	
	new_MPS.push_back(psi(1)*(itev)(1)*first_combiner);
	new_link_indices.push_back(first_link_index);

	for(int i = 2; i <= num_sites; i++){
		auto MPO_left_link = itensor::leftLinkIndex(itev, i);
		auto MPS_left_link = itensor::leftLinkIndex(psi, i);
		auto [left_combiner, left_combined_index] = itensor::combiner(MPO_left_link, MPS_left_link);
		if(i == num_sites){
			new_MPS.push_back(psi(num_sites)*(itev)(num_sites)*left_combiner*itensor::delta(left_combined_index, new_link_indices[num_sites-2]));
			break;
		}
		auto MPO_right_link = itensor::rightLinkIndex(itev, i);
		auto MPS_right_link = itensor::rightLinkIndex(psi, i);
		auto [right_combiner, right_combined_index] = itensor::combiner(itensor::IndexSet(MPO_right_link, MPS_right_link), {"Tags=","Link,l="+std::to_string(i)});
		new_MPS.push_back(psi(i)*(itev)(i)*left_combiner*right_combiner*itensor::delta(left_combined_index, new_link_indices[i-2]));
		new_link_indices.push_back(right_combined_index);
	}
	for(int i = 1; i <= num_sites; i++){
		psi.ref(i) = new_MPS[i-1];
	}
	psi.replaceLinkInds(itensor::IndexSet(new_link_indices));
	psi.replaceSiteInds(itensor::noPrime(itensor::siteInds(psi)));
}

int main(int argc, char *argv[]){
	int target_argc = 2;
	if(argc != target_argc){
		std::cerr << "Please provide an input file" << std::endl;
		return 1;
	}

	//Take inputs
	std::ifstream input_file_reader(argv[1]);
	if(!input_file_reader.is_open()){
		std::cerr <<"FILENAME " << argv[1] << " NOT FOUND" << endl;
		return 2;
	}

	InputClass input;
	input.Read(input_file_reader);

	int num_sites = input.getInteger("num_sites");
	int max_bd = input.getInteger("max_bond_dimension");
	double Jz = input.testDouble("Jz",1.0);
	double h = input.testDouble("external_field",0.0);
	int num_iterations = input.getInteger("num_iterations");
	std::string out_file_name = input.GetVariable("out_file");
	std::string mps_file_name = input.GetVariable("mps_file");
	double J2 = input.testDouble("J2", 0.0);
	double J3 = input.testDouble("J3", 0.0);
	std::string bond_list_file_name = input.testString("bond_list_file_name", "");
	std::string hamiltonian_type = input.testString("hamiltonian_type", "XXZ");
	

	std::cerr << "Read input files" << endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});

	

	OperatorMaker opm(sites);
	itensor::AutoMPO ampo;

	if(hamiltonian_type == "J1J3"){
		ampo = opm.J1J3Hamiltonian(J2, J3);
	}
	if(hamiltonian_type == "XC8Lattice"){
		ampo = opm.Lattice(J2, bond_list_file_name, num_sites, 12);
	}
	if(hamiltonian_type == "XXZ"){
		ampo = opm.XXZHamiltonian(Jz, h);
	}

	itensor::MPO H = itensor::toMPO(ampo);

	auto avg_Sz_ampo = opm.AverageSz();
	itensor::MPO avg_Sz = itensor::toMPO(avg_Sz_ampo);

	auto psi = itensor::randomMPS(sites);
	itensor::MPS trial_wavefunction(psi);
	auto sweeps = itensor::Sweeps(1);
	sweeps.maxdim() = max_bd;

	std::ofstream out_file(out_file_name);
	vector<double> energies;
	vector<double> bond_dimensions;
	vector<double> fidelities;
	vector<double> mes;

	for(int iteration = 0; iteration < num_iterations; iteration++){
		auto [energy, new_psi] = itensor::dmrg(H, psi, sweeps, {"Silent", true});
		energy = energy/num_sites;
		energies.push_back(energy);
		bond_dimensions.push_back(itensor::maxLinkDim(new_psi));
		double fidelity = itensor::inner(trial_wavefunction, new_psi);
		fidelities.push_back(fidelity);
		double me = itensor::inner(trial_wavefunction, H, new_psi)/num_sites;
		mes.push_back(me);
		psi = new_psi;
		std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has energy " << energy << " and estimated energy " << me/fidelity <<  std::endl;
	}
	itensor::writeToFile(mps_file_name, psi);


	psi /= std::sqrt(itensor::inner(psi, psi));
	double gs_energy = itensor::inner(psi, H, psi);
	vector<double> taus;
	vector<double> errors;
	std::cerr << "Ground state has energy " << gs_energy << " and squared norm " << itensor::inner(psi, psi) << std::endl;
	for(int iteration = 0; iteration < num_iterations; iteration ++){
		double tau_to_graph = (static_cast<double>(iteration)/num_iterations);
		tau_to_graph *= tau_to_graph;
		itensor::MPO itev = itensor::toExpH(ampo, tau_to_graph);
		itensor::MPS new_psi(psi);
		apply_MPO_no_truncation(itev, new_psi);
		double error = (itensor::inner(psi, new_psi) - std::exp(-tau_to_graph*gs_energy)*itensor::inner(psi, psi))/num_sites;
		std::cerr << "Tau = " << tau_to_graph << " has error " << error << std::endl;
		taus.push_back(tau_to_graph);
		errors.push_back(error);
	}

	out_file << "#MAX_BOND_DIMENSION:\n" << max_bd;
	out_file << "\n#NUM_SWEEPS:\n" << num_iterations;
	out_file << "\n#NUM_SITES:\n" << num_sites;
	out_file << "\n#HAMILTONIAN_TYPE:\n" << hamiltonian_type;
	out_file << "\n#HAMILTONIAN_PARAMETERS:\n";
	if(hamiltonian_type == "J1J3"){
		out_file << J2 << " " << J3;
	}
	else{
		out_file << Jz << " " << h;
	}
	out_file << "\n#ENERGIES:";
	for(double energy:energies){
		out_file << "\n" << energy;
	}
	out_file << "\n#FIDELITY:";
	for(double fidelity:fidelities){
		out_file << "\n" << fidelity;
	}
	out_file << "\n#MATRIX_ELEMENTS:";
	for(double me:mes){
		out_file << "\n" << me;
	}
	out_file << "\n#BOND_DIMENSIONS:";
	for(double bd:bond_dimensions){
		out_file << "\n" << bd;
	}
	out_file << "\n#TAUS:";
	for(double tau:taus){
		out_file << "\n" << tau;
	}
	out_file << "\n#ERRORS:";
	for(double error:errors){
		out_file << "\n" << error;
	}
	
	out_file.close();
	itensor::PrintData(psi);
	//itensor::writeToFile(mps_file_name, psi);
}