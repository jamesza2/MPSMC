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

	std::cerr << "Read input files" << endl;

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});
	OperatorMaker opm(sites);
	auto ampo = opm.XXZHamiltonian(Jz, h);
	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);

	//std::cerr << "Generating random spin configuration..." << endl;
	//Generate a random configuration
	/*itensor::InitState random_config_init(sites, "Up");
	std::mt19937 generator(std::time(NULL));
	std::string sequence = "";
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
	std::cerr << "Configuration: " << sequence << endl;
	itensor::MPS random_config(random_config_init);*/

	std::cerr << "Constructing initial High-BD state..." << endl;

	ThermalSystem sys(sites, itev, tau, 1000000, truncated_bds[0]);

	//Repeatedly applying itev to psi in order to create an MPS with >max_bd bond dimension
	int num_setup_iterations = 100;
	if(input.IsVariable("num_setup_iterations")){
		num_setup_iterations = input.getInteger("num_setup_iterations");
	}
	for(int i = 0; i < num_setup_iterations; i++){
		sys.iterate_single();
	}
	while(itensor::maxLinkDim(sys.psi) <= max_bd){
		sys.iterate_single_no_truncation();
	}

	itensor::MPS original_psi = sys.copy_state();

	if(input.IsVariable("original_MPS_file")){
		itensor::writeToFile(input.GetVariable("original_MPS_file"), original_psi);
	}
	
	itensor::InitState random_config_init(sites, "Up");
	if((method == "random")||(method == "Random")){
		std::cerr << "Finding a random configuration..." << endl;
		std::mt19937 generator(std::time(NULL));
		std::string sequence = "";
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
		std::cerr << "Configuration: " << sequence << endl;
	}
	else{
		std::cerr << "Finding a large overlap configuration..." << endl;
		auto Sz1_op = itensor::toMPO(opm.SingleSiteSz(1));
		double Sz1 = sys.expectation_value(Sz1_op);
		bool Sz1_up = true; //Determines whether the first site in the selected configuration should be up or down
		std::string sequence = "";
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
			double Sc1j = sys.expectation_value(Sc1j_op);
			if(Sc1j*Sz1 < 0){
				sequence += "D";
				random_config_init.set(j,"Dn");
			}
			else{
				sequence += "U";
			}
		}
		std::cerr << "Configuration: " << sequence << endl;
	}
	
	itensor::MPS random_config(random_config_init);
	

	//double original_overlap = std::real(itensor::innerC(original_psi, random_config))/std::sqrt(std::abs(itensor::innerC(original_psi, original_psi)*itensor::innerC(random_config, random_config)));
	double original_overlap = sys.overlap(random_config);
	std::cerr << "Original overlap: " << original_overlap << endl;
	std::cerr << "Creating truncated overlaps..." << endl;
	vector<double> average_overlaps;
	time_t start_time = std::time(NULL);
	vector<double> average_max_truncation;
	vector<double> average_average_truncation;
	for(int i = 0; i < truncated_bds.size(); i++){
		int num_selections = truncated_bds[i];
		sys.set_truncated_bd(num_selections);
		vector<double> overlaps;
		vector<double> truncations_max;
		vector<double> truncations_average;
		for(int i = 0; i < num_truncations; i++){
			sys.set_MPS(original_psi);
			sys.truncate();
			double ov = sys.overlap(random_config);
			overlaps.push_back(ov);
			truncations_max.push_back(((float)(sys.get_max_bd()))/itensor::maxLinkDim(original_psi));
			truncations_average.push_back(sys.get_avg_bd()/itensor::averageLinkDim(original_psi));
			std::cerr << "Overlap with configuration: " << ov << " | Old Max BD: " << itensor::maxLinkDim(original_psi) << " | New Max BD: " << sys.get_max_bd() << endl;
			//double fid = sys.overlap(original_psi);
			//std::cerr << "Fidelity with original state: " << fid << "(" << i+1 << "/" << num_truncations << ", " << std::difftime(time(NULL), start_time) << "s)" <<  endl;
			
		}
		double average_overlap = sum(overlaps)/num_truncations;
		average_overlaps.push_back(average_overlap);
		average_max_truncation.push_back(sum(truncations_max)/truncations_max.size());
		average_average_truncation.push_back(sum(truncations_average)/truncations_max.size());
		std::cerr << "Original overlap: " << original_overlap << " Final average overlap: " << average_overlap << " (" << i+1 << "/" << truncated_bds.size() << ", " << std::difftime(time(NULL), start_time) << "s)" <<  endl;
		start_time = time(NULL);
	}
	

	

	ofstream out_file(out_file_name);
	out_file << "#ORIGINAL_OVERLAP:\n" << original_overlap << "\n";
	out_file << "#NUM_SELECTIONS:";
	for(int num_selections : truncated_bds){
		out_file << "\n" << num_selections;
	}
	out_file << "\n#TRUNCATED_OVERLAPS:";
	for(double average_overlap : average_overlaps){
		out_file << "\n" << average_overlap;
	}
	out_file << "\n#AVERAGE_MAX_TRUNCATION:";
	for(double average_truncation : average_max_truncation){
		out_file << "\n" << average_truncation;
	}
	out_file << "\n#AVERAGE_AVERAGE_TRUNCATION:";
	for(double average_truncation : average_average_truncation){
		out_file << "\n" << average_truncation;
	}
	out_file << "\n#ORIGINAL_BOND_DIMENSION:\n" << itensor::maxLinkDim(original_psi);
	out_file.close();

}