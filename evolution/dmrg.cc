#include "itensor/all.h"
#include "../headers/input.h"
#include "../headers/operators.h"
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
	double Jz = input.getDouble("Jz");
	double h = input.getDouble("external_field");
	int num_iterations = input.getInteger("num_iterations");
	std::string out_file_name = input.GetVariable("out_file");
	

	std::cerr << "Read input files" << endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});

	OperatorMaker opm(sites);
	auto ampo = opm.XXZHamiltonian(Jz, h);
	itensor::MPO H = itensor::toMPO(ampo);

	auto avg_Sz_ampo = opm.AverageSz();
	itensor::MPO avg_Sz = itensor::toMPO(avg_Sz_ampo);

	auto psi = itensor::randomMPS(sites);
	auto sweeps = itensor::Sweeps(1);
	sweeps.maxdim() = max_bd;

	std::ofstream out_file(out_file_name);
	vector<double> energies;
	vector<double> bond_dimensions;

	for(int iteration = 0; iteration < num_iterations; iteration++){
		auto [energy, new_psi] = itensor::dmrg(H, psi, sweeps, {"Quiet", true});
		energy = energy/num_sites;
		energies.push_back(energy);
		bond_dimensions.push_back(itensor::maxLinkDim(new_psi));
		psi = new_psi;
		std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has energy " << energy << endl;
	}

	out_file << "#MAX_BOND_DIMENSION:\n" << max_bd;
	out_file << "\n#JZ:\n" << Jz;
	out_file << "\n#EXTERNAL_FIELD:\n" << h;
	out_file << "\n#NUM_SWEEPS:\n" << num_iterations;
	out_file << "\n#NUM_SITES:\n" << num_sites;
	out_file << "\n#ENERGIES:";
	for(double energy:energies){
		out_file << "\n" << energy;
	}
	out_file << "\n#BOND_DIMENSIONS:";
	for(double bd:bond_dimensions){
		out_file << "\n" << bd;
	}
	out_file.close();
	//itensor::writeToFile(mps_file_name, psi);
}