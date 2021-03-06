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

/*void fix_sites(itensor::MPS &psi, itensor::SiteSet &sites){
	for(int i = 1; i <= itensor::length(psi); i++){
		psi.ref(i) *= itensor::delta(itensor::siteIndex(psi, i), )
	}
}*/

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
	double Jz = input.testDouble("Jz", 1.0);
	double h = input.testDouble("external_field", 0.0);
	double J2 = input.testDouble("J2", 0.0);
	double J3 = input.testDouble("J3", 0.0);
	std::string hamiltonian_type = input.testString("hamiltonian_type", "XXZ");
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
	bool false_gs = input.testBool("false_gs", false);
	int false_gs_bond_dimension = input.testInteger("false_gs_bond_dimension", 400);
	std::string true_gs_file = input.testString("true_ground_state_file", "");
	std::string bond_list_file_name = input.testString("bond_list_file", "");
	double singular_value_sum_threshhold = input.testDouble("singular_value_sum_threshhold", 0);
	double random_singular_value_weight = input.testDouble("random_singular_value_weight", 1.0);
	bool record_truncation_fidelities = input.testBool("record_truncation_fidelities", false);
	if(false_gs){
		true_gs_file = "";
	}
	vector<int> rearranged_sites = input.testVectorInt("rearranged_sites", vector<int>(0));
	std::string trial_energy_calculation_mode = input.testString("trial_energy_calculation_mode", "NORMAL");

	std::streambuf *coutbuf = std::cerr.rdbuf();
	std::ofstream log_file(log_file_name);
	if(log_file_name != ""){
		std::cerr.rdbuf(log_file.rdbuf());
	}
	

	std::cerr << "Read input files" << std::endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});

	itensor::MPS trial(sites);
	if(trial_wavefunction_file_name != ""){
		trial = itensor::readFromFile<itensor::MPS>(trial_wavefunction_file_name, sites);
		trial.replaceSiteInds(itensor::inds(sites));
		//read_from_file(sites, trial_wavefunction_file_name, trial);
	}

	itensor::MPS true_gs(sites);
	if(true_gs_file != ""){
		true_gs = itensor::readFromFile<itensor::MPS>(true_gs_file, sites);
	}


	//std::cerr << "Read true gs and trial states" << std::endl;

	OperatorMaker opm(sites);
	itensor::AutoMPO ampo;
	if(hamiltonian_type == "J1J3"){
		ampo = opm.J1J3Hamiltonian(J2, J3);
	}
	if(hamiltonian_type == "XC8Lattice")
	{
		ampo = opm.Lattice(J2, bond_list_file_name, num_sites, 12);
	}
	if(hamiltonian_type == "XXZ"){
		ampo = opm.XXZHamiltonian(Jz, h);
	}
	
	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);

	//Print(H);
	//Print(itev);

	auto avg_Sz_ampo = opm.AverageSz();
	itensor::MPO avg_Sz = itensor::toMPO(avg_Sz_ampo);



	ThermalWalkers tw(sites, itev, H, tau, max_bd, truncated_bd, num_walkers, num_max_walkers);
	if(trial_bd != 0){
		itensor::MPS random_trial = randomMPS::randomMPS(sites, trial_bd, trial_correlation_length);
		random_trial.normalize();
		tw.set_trial_wavefunction(random_trial);
		tw.set_MPS(random_trial, 0);
	}
	tw.set_random_singular_value_weight(random_singular_value_weight);
	tw.set_kept_singular_values(kept_singular_values);
	tw.set_record_truncation_fidelities(record_truncation_fidelities);
	tw.fixed_node = fixed_node;
	tw.verbose = verbose;
	tw.singular_value_sum_threshhold = singular_value_sum_threshhold;
	
	tw.set_trial_energy_calculation_mode(trial_energy_calculation_mode);

	if(trial_wavefunction_file_name != ""){

		tw.set_trial_wavefunction(trial);
	}

	tw.set_fixed_node_wavefunction(tw.trial_wavefunction);

	if(fn_wavefunction_file != ""){
		itensor::MPS fn = itensor::readFromFile<itensor::MPS>(fn_wavefunction_file, sites);
		fn.replaceSiteInds(itensor::inds(sites));
		//itensor::MPS fn(sites);
		//read_from_file(sites, fn_wavefunction_file, fn);
		tw.set_fixed_node_wavefunction(fn);
	}

	if(starting_wavefunction_file != ""){
		auto sw = itensor::readFromFile<itensor::MPS>(starting_wavefunction_file,sites);
		sw.replaceSiteInds(itensor::inds(sites));
		/*itensor::MPS sw(sites);
		std::cout << "Reading starting wavefunction from file" << std::endl;
		read_from_file(sites, starting_wavefunction_file, sw);
		std::cout << "Setting starting walker to starting wavefunction..." << std::endl;*/
		tw.set_MPS(sw, 0);
	}

	if(false_gs){
		true_gs = itensor::MPS(tw.walkers[0]);
	}

	std::cerr << "Trial wavefunction native energy: " << itensor::innerC(tw.trial_wavefunction, H, tw.trial_wavefunction)/itensor::innerC(tw.trial_wavefunction, tw.trial_wavefunction) << std::endl;

	std::ofstream out_file(out_file_name);
	//out_file << "Energy|Bond Dimension|Max Sz" << endl;
	vector<double> average_energies;
	vector<vector<double>> walker_energies;
	vector<vector<double>> walker_weights;
	vector<vector<double>> walker_overlaps;
	vector<vector<double>> true_gs_overlaps;
	vector<int> num_current_walkers;
	vector<vector<int>> bds;
	vector<double> trial_energies;
	vector<double> entanglement_entropies;
	vector<double> first_state_energies;
	vector<vector<double>> truncation_fidelities;
	for(int iteration = 0; iteration < num_iterations; iteration++){
		time_t start_time = time(NULL);
		trial_energies.push_back(tw.trial_energy);
		if(iteration > 0){
			tw.iterate_single();
		}
		double energy = tw.expectation_value(H);

		if(false_gs){
			true_gs = itensor::applyMPO(itev, true_gs, {"Maxdim=", false_gs_bond_dimension});
		}
		
		vector<double> fidelities = tw.truncation_fidelities;
		vector<double> energies = tw.expectation_values(H);
		vector<double> weights = tw.get_weights();
		vector<double> overlaps = tw.weighted_overlaps(tw.trial_wavefunction);
		//std::cerr << "Measuring overlap with true ground state: " << std::endl;
		//Print(true_gs);
		//Print(tw.walkers[0]);
		if(true_gs_file != ""){
			//std::cout << "Calculating true GS overlaps..." << std::endl;
			vector<double> tgs_overlaps = tw.weighted_overlaps(true_gs);
			true_gs_overlaps.push_back(tgs_overlaps);
		}
		
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
		truncation_fidelities.push_back(fidelities);
		num_current_walkers.push_back(tw.weights.size());
		entanglement_entropies.push_back(tw.average_entanglement_entropy(num_sites/2));
		bds.push_back(tw.get_bds());
		std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has average weighted energy " << energy << " among " << weights.size() << " walkers (" << difftime(time(NULL), start_time) << "s)" << std::endl;
		
		double first_state_energy = itensor::inner(tw.walkers[0], H, tw.walkers[0])/(num_sites*tw.weights[0]*tw.weights[0]);
		std::cerr << "First state energy after truncation: " << first_state_energy << std::endl;
		first_state_energies.push_back(first_state_energy);
		//std::cerr << "Trial wavefunction native energy: " << itensor::inner(tw.trial_wavefunction, H, tw.trial_wavefunction)/(num_sites*itensor::inner(tw.trial_wavefunction, tw.trial_wavefunction)) << std::endl;
		std::cout << "Iteration " << iteration+1 << "/" << num_iterations << " complete (" << difftime(time(NULL), start_time) << "s)\r";
		std::cout << std::flush;
		start_time = time(NULL);
		
		//out_file << energy << "|" << bond_dimension << "|" << avg_Sz_val << endl;
	}

	std::cerr << tw.print_times() << std::endl;

	if(log_file_name != ""){
		std::cerr.rdbuf(coutbuf);
	}
	log_file.close();
	out_file << "#NUM_EXPECTED_WALKERS:\n" << num_walkers;
	out_file << "\n#NUM_ITERATIONS:\n" << num_iterations;
	out_file << "\n#TAU:\n" << tau;
	out_file << "\n#MAX_BD:\n" << max_bd;
	out_file << "\n#TRUNCATED_BD:\n" << truncated_bd;
	out_file << "\n#TRIAL_ENERGY_CALCULATION_MODE:\n" << trial_energy_calculation_mode;
	out_file << "\n#HAMILTONIAN_TYPE:\n" << hamiltonian_type;
	out_file << "\n#RANDOM_SINGULAR_VALUE_WEIGHT\n" << random_singular_value_weight;
	out_file << "\n#HAMILTONIAN_PARAMETERS:\n";
	if(hamiltonian_type == "J1J3"){
		out_file << J2 << " " << J3;
	}
	else{
		out_file << Jz << " " << h;
	}
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
	if(true_gs_file != ""){
		out_file << "\n#TRUE_GROUND_STATE_OVERLAPS:";
		for(vector<double> true_gs_overlap_vector : true_gs_overlaps){
			out_file << "\n";
			for(double true_gs_overlap : true_gs_overlap_vector){
				out_file << true_gs_overlap << " ";
			}
		}
	}

	if(record_truncation_fidelities){
		out_file << "\n#TRUNCATION_FIDELITIES:";
		for(vector<double> truncation_fidelities_at_step : truncation_fidelities){
			out_file << "\n";
			for(double fidelity : truncation_fidelities_at_step){
				out_file << fidelity << " ";
			}
		}
	}
	
	out_file << "\n#FIRST_STATE_ENERGIES:\n";
	for(double fse : first_state_energies){
		out_file << fse << " ";
	}

	out_file << "\n#BOND_DIMENSIONS:";
	for(vector<int> bond_dimension_vector : bds){
		out_file << "\n";
		for(int bd : bond_dimension_vector){
			out_file << bd << " ";
		}
	}

	out_file << "\n" << tw.walker_path;



	//itensor::writeToFile(mps_file_name, psi);
}