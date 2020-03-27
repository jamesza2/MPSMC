#include "itensor/all.h"
#include "../headers/input.h"
#include "../headers/operators.h"
#include "../headers/thermal_system.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>


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
	int truncated_bd = input.getInteger("truncated_bd");
	double tau = input.getDouble("tau");
	double Jz = input.getDouble("Jz");
	double h = input.getDouble("external_field");
	int num_iterations = input.getInteger("num_iterations");
	double threshhold = input.getDouble("threshhold");
	//std::string mps_file_name = input.GetVariable("mps_file_name");
	std::string out_file_name = input.GetVariable("out_file");
	

	std::cerr << "Read input files" << endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});

	OperatorMaker opm(sites);
	auto ampo = opm.XXZHamiltonian(Jz, h);
	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);

	auto avg_Sz_ampo = opm.AverageSz();
	itensor::MPO avg_Sz = itensor::toMPO(avg_Sz_ampo);


	ThermalSystem sys(sites, itev, tau, max_bd, truncated_bd);


	std::ofstream out_file(out_file_name);
	out_file << "Energy|Bond Dimension|Max Sz" << endl;
	for(int iteration = 0; iteration < num_iterations; iteration++){
		sys.iterate_simple(threshhold);
		double energy = sys.expectation_value(H);
		energy = energy/num_sites;
		std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has energy " << energy << endl;
		int bond_dimension = itensor::maxLinkDim(sys.psi);
		double avg_Sz_val = sys.expectation_value(avg_Sz);
		
		out_file << energy << "|" << bond_dimension << "|" << avg_Sz_val << endl;
	}
	out_file.close();
	//itensor::writeToFile(mps_file_name, psi);
}