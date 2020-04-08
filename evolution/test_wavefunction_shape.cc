#include "../headers/thermal_system.h"
#include "../headers/operators.h"
#include "../headers/input.h"
#include <ctime>
#include <cmath>
#include <complex>
#include <random>

double sum(std::vector<double> v){
	double s = 0;
	for(double elem:v){
		s += elem;
	}
	return s;
}

int main(int argc, char*argv[]){
	
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
	vector<int> truncated_bds = input.getVectorInteger("truncated_bds");
	double tau = input.getDouble("tau");
	double Jz = input.getDouble("Jz");
	double h = input.getDouble("external_field");
	int num_truncations = input.getInteger("num_truncations");
	std::string method = input.GetVariable("configuration_selection");
	std::string out_file_name = input.GetVariable("out_file");
	bool keep_weight = input.testBool("keep_weight", false);
	bool metropolis_sampling = input.testBool("metropolis_sampling", false);
	int num_metropolis_setup_steps = input.testInt("num_metropolis_setup_steps");
	int num_metropolis_sample_steps = input.testInt("num_metropolis_sample_steps");
	int num_configurations = input.getInteger("num_configurations");
	bool specific_sites = input.testBool("specific_sites", false);
	vector<int> site;
	if(specific_sites){
		site = input.getVectorInteger("site");
	}
	else{
		for(int i = 1; i < num_sites; i++){
			site.push_back(i);
		}
	}

	std::cerr << "Read input files" << endl;

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});
	OperatorMaker opm(sites);
	auto ampo = opm.XXZHamiltonian(Jz, h);
	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);

	std::cerr << "Constructing initial High-BD state..." << endl;

	ThermalSystem sys(sites, itev, tau, max_bd, 100);
	sys.keep_weight = keep_weight;

	//Repeatedly applying itev to psi in order to create an MPS with >max_bd bond dimension
	int num_setup_iterations = 100;
	if(input.IsVariable("num_setup_iterations")){
		num_setup_iterations = input.getInteger("num_setup_iterations");
	}
	for(int i = 0; i < num_setup_iterations; i++){
		sys.iterate_single();
	}

	sys.set_truncated_bd(truncated_bds[0]);

	while(itensor::maxLinkDim(sys.psi) <= max_bd){
		sys.iterate_single_no_truncation();
	}

	itensor::MPS original_psi = sys.copy_state();

	if(input.IsVariable("original_MPS_file")){
		itensor::writeToFile(input.GetVariable("original_MPS_file"), original_psi);
	}
	vector<double> original_overlaps;
	vector<itensor::MPS> configs;
	for(int configuration_index = 0; configuration_index < num_configurations; configuration_index ++){
		itensor::InitState random_config_init(sites, "Up");
		std::string sequence = "";
		if((method == "random")||(method == "Random")){
			//std::cerr << "Finding a random configuration..." << endl;
			std::mt19937 generator(std::time(NULL));
			for(int i = 1; i <= num_sites; i++){
				int random_value = generator();
				if(random_value % 2 == 1){
					random_config_init.set(i, "Dn");
					sequence += "D";
				}
				else{
					sequence += "U";
				}
			}
		}
		else{
			//std::cerr << "Finding a large overlap configuration..." << endl;
			auto Sz1_op = itensor::toMPO(opm.SingleSiteSz(1));
			double Sz1 = sys.expectation_value(Sz1_op);
			bool Sz1_up = true; //Determines whether the first site in the selected configuration should be up or down
			double random_sequence = sys.random_double() - 0.5;
			Sz1 += random_sequence;
			if(Sz1 < 0){
				random_config_init.set(1,"Dn");
				sequence += "D";
				Sz1_up = false;
			}
			else{
				sequence += "U";
			}


			
			for(int j = 2; j <= num_sites; j++){
				auto Sc1j_op = itensor::toMPO(opm.SpinCorrelation(1,j));
				double Sc1j = sys.expectation_value(Sc1j_op) + std::pow(sys.random_double()*1.4 - 0.7,5);
				if(Sc1j*Sz1 < 0){
					sequence += "D";
					random_config_init.set(j,"Dn");
				}
				else{
					sequence += "U";
				}
			}
		}
		itensor::MPS random_config(random_config_init);
		configs.push_back(random_config);
		double ov = sys.overlap(random_config);
		original_overlaps.push_back(ov);
		std::cerr << "Configuration " << sequence << " has overlap " << ov << endl;
	}
	std::cerr << "Original overlap with first config: " << original_overlaps[0] << endl;
	std::cerr << "Creating truncated overlaps..." << endl;
	time_t start_time = std::time(NULL);

	vector<vector<vector<double>>> estimated_errors; //First index is BD, second is truncation attempt number, third is site index
	vector<vector<vector<double>>> measured_overlaps;

	for(int bd_index = 0; bd_index < truncated_bds.size(); bd_index++){
		int num_selections = truncated_bds[bd_index];
		sys.set_truncated_bd(num_selections);
		std::cerr << "Testing truncation to bond dimension " << num_selections << std::endl;
		vector<vector<double>> estimated_errors_at_bd;
		
		vector<vector<double>> measured_overlaps_at_bd;
		
		for(int trunc_index = 0; trunc_index < num_truncations; trunc_index++){
			sys.estimated_error = 1.0;
			sys.set_MPS(original_psi);
			vector<double> estimated_errors_at_trunc;
			
			vector<double> measured_overlaps_at_trunc;
			
			for(int site_index = 0; site_index < site.size(); site_index++){
				int site_number = site[site_index];
				if(metropolis_sampling){
					sys.truncate_metropolis_single_site(site_number, num_metropolis_setup_steps, num_metropolis_sample_steps);
				}
				else{
					sys.truncate_single_site(site_number);
				}
			}
			for(int config_index = 0; config_index < num_configurations; config_index ++){
				double ov = sys.overlap(configs[config_index]);
				double estimated_error = sys.estimated_error;
				measured_overlaps_at_trunc.push_back(ov);
				estimated_errors_at_trunc.push_back(estimated_error);
			}
			estimated_errors_at_bd.push_back(estimated_errors_at_trunc);
			measured_overlaps_at_bd.push_back(measured_overlaps_at_trunc);
			std::cerr << "Overlap with first configuration: " << measured_overlaps_at_trunc[0] << " | First Estimated Error: " << estimated_errors_at_trunc[0] << " | " << difftime(start_time, time(NULL)) << "s" << endl;
			
		}
		estimated_errors.push_back(estimated_errors_at_bd);
		measured_overlaps.push_back(measured_overlaps_at_bd);

		//start_time = time(NULL);
	}
	

	

	ofstream out_file(out_file_name);
	out_file << "#ORIGINAL_OVERLAPS:\n";
	for(double original_overlap : original_overlaps){
		out_file << original_overlap << " | ";
	}
	for(int truncated_bd_index = 0; truncated_bd_index < truncated_bds.size(); truncated_bd_index ++){
		int truncated_bd = truncated_bds[truncated_bd_index];
		vector<vector<double>> estimated_errors_at_bd = estimated_errors[truncated_bd_index];
		vector<vector<double>> measured_overlaps_at_bd = measured_overlaps[truncated_bd_index];
		out_file << "\n#ESTIMATED_ERROR | BOND_DIMENSION, " << truncated_bd;
		for(int trunc_index = 0; trunc_index < num_truncations; trunc_index ++){
			vector<double> estimated_errors_at_trunc = estimated_errors_at_bd[trunc_index];
			out_file << "\n";
			for(double estimated_error : estimated_errors_at_trunc){
				out_file << estimated_error << " | ";
			}
		}
		out_file << "\n#TRUNCATED_OVERLAP | BOND_DIMENSION, " << truncated_bd;
		for(int trunc_index = 0; trunc_index < num_truncations; trunc_index ++){
			vector<double> measured_overlaps_at_trunc = measured_overlaps_at_bd[trunc_index];
			out_file << "\n";
			for(double measured_overlap : measured_overlaps_at_trunc){
				out_file << measured_overlap << " | ";
			}
		}
	}
	out_file.close();

}