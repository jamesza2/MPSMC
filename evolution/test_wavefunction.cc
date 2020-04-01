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
	int truncated_bd = input.getInteger("truncated_bd");
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

	ThermalSystem sys(sites, itev, tau, max_bd, truncated_bd);


	int num_setup_iterations = 100;
	if(input.IsVariable("num_setup_iterations")){
		num_setup_iterations = input.getInteger("num_setup_iterations");
	}
	for(int i = 0; i < num_setup_iterations; i++){
		sys.iterate_single();
	}
	//Repeatedly applying itev to psi in order to create an MPS with >max_bd bond dimension
	while(itensor::maxLinkDim(sys.psi) <= max_bd){
		sys.iterate_single_no_truncation();
	}

	itensor::MPS original_psi = sys.copy_state();

	
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
	vector<double> overlaps;
	vector<double> error_corrected_overlaps;
	time_t start_time = std::time(NULL);
	for(int i = 0; i < num_truncations; i++){
		sys.estimated_error = 1.0;
		sys.set_MPS(original_psi);
		/*double acquired_estimated_error = 1.0;
		for(int site_index = 1; site_index < num_sites; site_index ++){
			sys.truncate_single_site();
			double site_overlap = sys.overlap(random_config);
			acquired_estimated_error = sys.estimated_error/acquired_estimated_error;
		}*/
		sys.truncate();
		double ov = sys.overlap(random_config);
		overlaps.push_back(ov);
		error_corrected_overlaps.push_back(ov/sys.estimated_error);
		std::cerr << "Overlap with configuration: " << ov << " | Estimated Error: " << sys.estimated_error << endl;
		double fid = sys.overlap(original_psi);
		std::cerr << "Fidelity with original state: " << fid << "(" << i+1 << "/" << num_truncations << ", " << std::difftime(time(NULL), start_time) << "s)" <<  endl;
		
		start_time = time(NULL);
	}

	std::cerr << "Original overlap: " << original_overlap << " Final average overlap: " << sum(overlaps)/num_truncations << endl;
	std::cerr << "Average error corrected overlap: " << sum(error_corrected_overlaps)/num_truncations << endl;

	ofstream out_file(out_file_name);
	out_file << "#ORIGINAL_OVERLAP:\n" << original_overlap << "\n";
	out_file << "#TRUNCATED_OVERLAPS:";
	for(double overlap : overlaps){
		out_file << "\n" << overlap;
	}
	out_file << "\n#ERROR_CORRECTED_OVERLAPS:";
	for(double overlap : error_corrected_overlaps){
		out_file << "\n" << overlap;
	}
	out_file.close();

}