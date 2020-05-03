#include "itensor/all.h"
#include "../headers/input.h"
#include "../headers/operators.h"
#include "../headers/thermal_walkers.h"
#include "../headers/random_mps.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>

/*vector<double> sort_by(vector<double> &values, vector<double> &sorter){
	if(sorter.size() != values.size()){

	}
	vector<double> sorted_values;
	for(double val : sorter){

	}
}*/

void read_from_file(itensor::SiteSet &sites, std::string mps_file_name, itensor::MPS &psi){
	itensor::readFromFile(mps_file_name, psi);
	auto index_set = itensor::siteInds(psi);
	vector<itensor::Index> indices;
	int num_sites = itensor::length(sites);
	for(int site = 0; site < num_sites; site++){
		indices.push_back(index_set(site+1));
	}
	sites = itensor::SpinHalf(indices);
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
	int truncated_bd = input.getInteger("truncated_bd");
	double tau = input.getDouble("tau");
	double Jz = input.getDouble("Jz");
	double h = input.getDouble("external_field");
	int num_iterations = input.getInteger("num_iterations");
	int num_walkers = input.getInteger("num_walkers");
	int num_max_walkers = input.testInteger("num_max_walkers", num_walkers);
	//std::string mps_file_name = input.GetVariable("mps_file_name");
	std::string out_file_name = input.GetVariable("out_file");
	int kept_singular_values = input.testInteger("kept_singular_values", 0);
	std::string trial_wavefunction_file_name = input.testString("trial_wavefunction_file", "");
	std::string log_file_name = input.testString("log_file", "");
	bool verbose = input.testBool("verbose", false);
	bool fixed_node = input.testBool("fixed_node", false);
	int trial_bd = input.testInteger("trial_bond_dimension", 0);
	double trial_correlation_length = input.testDouble("trial_correlation_length", 0.5);
	std::string fn_wavefunction_file = input.testString("fixed_node_wavefunction_file", "");
	std::string starting_wavefunction_file = input.testString("starting_wavefunction_file", "");
	std::streambuf *coutbuf = std::cerr.rdbuf();
	std::ofstream log_file(log_file_name);
	if(log_file_name != ""){
		std::cerr.rdbuf(log_file.rdbuf());
	}
	

	std::cerr << "Read input files" << endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});

	itensor::MPS trial(sites);
	if(trial_wavefunction_file_name != ""){
		trial = itensor::readFromFile<itensor::MPS>(trial_wavefunction_file_name, sites);
		//read_from_file(sites, trial_wavefunction_file_name, trial);
	}

	OperatorMaker opm(sites);
	auto ampo = opm.XXZHamiltonian(Jz, h);
	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);

	auto avg_Sz_ampo = opm.AverageSz();
	itensor::MPO avg_Sz = itensor::toMPO(avg_Sz_ampo);

	std::cout << "Creating thermalwalkers object..." << std::endl;


	ThermalWalkers tw(sites, itev, tau, max_bd, truncated_bd, num_walkers, num_max_walkers);
	if(trial_bd != 0){
		std::cout << "Creating random trial wavefunction..." << std::endl;
		itensor::MPS random_trial = randomMPS::randomMPS(sites, trial_bd, trial_correlation_length);
		random_trial.normalize();
		tw.set_trial_wavefunction(random_trial);
		tw.set_MPS(random_trial, 0);
	}
	tw.set_kept_singular_values(kept_singular_values);
	tw.fixed_node = fixed_node;
	tw.verbose = verbose;
	
	if(trial_wavefunction_file_name != ""){
		std::cout << "Setting trial wavefunction from file..." << std::endl;
		tw.set_trial_wavefunction(trial);
	}

	std::cerr << "Setting fixed node wavefunction to trial wavefunction..." << std::endl;
	tw.set_fixed_node_wavefunction(tw.trial_wavefunction);

	if(fn_wavefunction_file != ""){
		std::cout << "Setting fixed node wavefunction from file..." << std::endl;
		itensor::MPS fn = itensor::readFromFile<itensor::MPS>(fn_wavefunction_file, sites);
		//itensor::MPS fn(sites);
		//read_from_file(sites, fn_wavefunction_file, fn);
		tw.set_fixed_node_wavefunction(fn);
	}

	std::cerr<< "Creating starting wavefunction..." << std::endl;
	if(starting_wavefunction_file != ""){
		auto sw = itensor::readFromFile<itensor::MPS>(starting_wavefunction_file,sites);
		/*itensor::MPS sw(sites);
		std::cout << "Reading starting wavefunction from file" << std::endl;
		read_from_file(sites, starting_wavefunction_file, sw);
		std::cout << "Setting starting walker to starting wavefunction..." << std::endl;*/
		tw.set_MPS(sw, 0);
		std::cout << "Starting wavefunction has norm " << itensor::norm(tw.walkers[0]) << " and trial overlap " << itensor::inner(tw.walkers[0], tw.trial_wavefunction) << std::endl;
	}
	std::cout << "Starting ite..." << std::endl;

	std::ofstream out_file(out_file_name);
	//out_file << "Energy|Bond Dimension|Max Sz" << endl;
	vector<double> average_energies;
	vector<vector<double>> walker_energies;
	vector<vector<double>> walker_weights;
	vector<vector<double>> walker_overlaps;
	vector<int> num_current_walkers;
	vector<vector<int>> bds;
	vector<double> trial_energies;
	vector<double> entanglement_entropies;
	for(int iteration = 0; iteration < num_iterations; iteration++){
		time_t start_time = time(NULL);
		trial_energies.push_back(tw.trial_energy);
		if(iteration > 0){
			tw.iterate_single();
		}
		std::cout << "Computing expectation value..." << std::endl;
		Print(tw.trial_wavefunction);
		Print(tw.walkers[0]);
		Print(H);
		Print(sites);
		double energy = tw.expectation_value(H);

		vector<double> energies = tw.expectation_values(H);
		vector<double> weights = tw.get_weights();
		std::cout << "Computing overlaps..." << std::endl;
		vector<double> overlaps = tw.weighted_overlaps(tw.trial_wavefunction);
		/*if(trial_wavefunction_file_name != ""){
			overlaps = tw.weighted_overlaps(trial);
		}
		else{
			overlaps = tw.weighted_overlaps(tw.trial_wavefunction);
		}*/
		//tw.recalculate_trial_energy(energy);
		walker_energies.push_back(energies);
		walker_weights.push_back(weights);
		walker_overlaps.push_back(overlaps);
		average_energies.push_back(energy);
		num_current_walkers.push_back(tw.weights.size());
		std::cout << "Computing entanglement entropies..." << std::endl;
		entanglement_entropies.push_back(tw.average_entanglement_entropy(num_sites/2));
		bds.push_back(tw.get_bds());
		std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has average weighted energy " << energy << " among " << weights.size() << " walkers (" << difftime(time(NULL), start_time) << "s)" << std::endl;
		std::cerr << "Native energy of first state: " << itensor::inner(tw.walkers[0], H, tw.walkers[0])/(num_sites*tw.weights[0]*tw.weights[0]) << std::endl;
		start_time = time(NULL);
		
		//out_file << energy << "|" << bond_dimension << "|" << avg_Sz_val << endl;
	}

	if(log_file_name != ""){
		std::cerr.rdbuf(coutbuf);
	}
	log_file.close();
	out_file << "#NUM_EXPECTED_WALKERS:\n" << num_walkers;
	out_file << "\n#NUM_ITERATIONS:\n" << num_iterations;
	out_file << "\n#TAU:\n" << tau;
	out_file << "\n#MAX_BD:\n" << max_bd;
	out_file << "\n#TRUNCATED_BD:\n" << truncated_bd;
	out_file << "\n#AVERAGE_ENERGIES:\n";
	for(double energy : average_energies){
		out_file << energy << " ";
	}
	out_file << "\n#AVERAGE_ENTROPIES:\n";
	for(double entropy : entanglement_entropies){
		out_file << entropy << " ";
	}
	out_file << "\n#NUM_WALKERS:\n";
	for(double nw : num_current_walkers){
		out_file << nw << " ";
	}
	out_file << "\n#TRIAL_ENERGIES:\n";
	for(double energy : trial_energies){
		out_file << energy << " ";
	}
	out_file << "\n#WALKER_ENERGIES:";
	for(vector<double> walker_energy_vector : walker_energies){
		out_file << "\n";
		for(double walker_energy : walker_energy_vector){
			out_file << walker_energy << " ";
		}
	}
	out_file << "\n#WALKER_OVERLAPS:";
	for(vector<double> walker_overlap_vector : walker_overlaps){
		out_file << "\n";
		for(double walker_overlap : walker_overlap_vector){
			out_file << walker_overlap << " ";
		}
	}
	out_file << "\n#WALKER_WEIGHTS:";
	for(vector<double> walker_weight_vector : walker_weights){
		out_file << "\n";
		for(double walker_weight : walker_weight_vector){
			out_file << walker_weight << " ";
		}
	}

	out_file << "\n#BOND_DIMENSIONS:";
	for(vector<int> bond_dimension_vector : bds){
		out_file << "\n";
		for(int bd : bond_dimension_vector){
			out_file << bd << " ";
		}
	}



	//itensor::writeToFile(mps_file_name, psi);
}