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
	int trial_bd = input.getInteger("trial_bond_dimension");
	double trial_correlation_length = input.getDouble("trial_correlation_length");
	std::string mps_file_name = input.GetVariable("mps_file");
	

	std::cerr << "Read input files" << endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});

	itensor::MPS random_trial = randomMPS::randomMPS(sites, trial_bd, trial_correlation_length);
	random_trial.normalize();
	itensor::writeToFile(mps_file_name, random_trial);
	
	//itensor::writeToFile(mps_file_name, psi);
}