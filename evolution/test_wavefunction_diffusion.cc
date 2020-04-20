#include "../headers/thermal_walkers.h"
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
	int num_desired_walkers = input.getInteger("num_walkers");
	int num_max_walkers = input.testInteger("num_max_walkers", num_desired_walkers);
	std::string trial_wavefunction_file_name = input.testString("trial_wavefunction_file", "");


	std::cerr << "Read input files" << endl;

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});
	

	std::cerr << "Constructing initial High-BD state..." << endl;

	
	itensor::MPS trial(sites);
	if(trial_wavefunction_file_name != ""){
		
		if(trial_wavefunction_file_name != ""){
			read_from_file(sites, trial_wavefunction_file_name, trial);
		}
		
	}

	OperatorMaker opm(sites);
	auto ampo = opm.XXZHamiltonian(Jz, h);
	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);
	ThermalWalkers tw(sites, itev, tau, max_bd, truncated_bd, num_desired_walkers, num_max_walkers);
	if(trial_wavefunction_file_name != ""){
		tw.set_MPS(trial,0);
	}
	else{
		int num_setup_iterations = 100;
		if(input.IsVariable("num_setup_iterations")){
			num_setup_iterations = input.getInteger("num_setup_iterations");
		}
		for(int i = 0; i < num_setup_iterations; i++){
			tw.iterate_single();
			tw.recalculate_trial_energy(tw.average_walker_energy(H));
		}
		//Repeatedly applying itev to psi in order to create an MPS with >max_bd bond dimension
		while(tw.get_max_bd() <= max_bd){
			tw.iterate_single_no_truncation();
			tw.recalculate_trial_energy(tw.average_walker_energy(H));
		}

	}
		//tw.shave(1);

	auto original_psis = tw.get_all_MPS();

	
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
		double Sz1 = tw.expectation_value(Sz1_op);
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
			double Sc1j = tw.expectation_value(Sc1j_op);
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
	tw.set_trial_wavefunction(random_config);
	
	//double original_overlap = std::real(itensor::innerC(original_psi, random_config))/std::sqrt(std::abs(itensor::innerC(original_psi, original_psi)*itensor::innerC(random_config, random_config)));
	double original_overlap = tw.overlap(random_config);
	vector<double> original_overlaps = tw.overlaps(random_config);
	std::cerr << "Original number of walkers: " << tw.weights.size() << std::endl;
	std::cerr << "Original overlap: " << original_overlap << std::endl;
	std::cerr << "Creating truncated overlaps..." << std::endl;
	vector<double> overlaps;
	vector<int> num_walkers;
	vector<vector<double>> individual_overlaps;
	vector<vector<double>> walker_weights;
	vector<vector<double>> individual_mes;
	vector<vector<double>> individual_fidelities;

	time_t start_time = std::time(NULL);
	for(int i = 0; i < num_truncations; i++){
		tw.set_all_MPS(original_psis);
		tw.process();
		double ov = tw.overlap(random_config);
		overlaps.push_back(ov);
		vector<double> ovs = tw.overlaps(random_config);
		vector<double> fids = ovs;
		vector<double> mes;
		if(trial_wavefunction_file_name != ""){
			fids = tw.weighted_overlaps(random_config);
			mes = tw.expectation_values(H);
		}
		individual_mes.push_back(mes);
		individual_fidelities.push_back(fids);
		individual_overlaps.push_back(ovs);
		num_walkers.push_back(tw.weights.size());
		vector<double> ww = tw.weights;
		walker_weights.push_back(ww);
		std::cerr << "Overlap with configuration: " << ov << std::endl;
		std::cerr << "First BD: " << tw.get_bds()[0] << std::endl;
		std::cerr << " (" << i+1 << "/" << num_truncations << ", " << std::difftime(time(NULL), start_time) << "s)" <<  std::endl;
		start_time = time(NULL);
	}

	std::cerr << "Original overlap: " << original_overlap << " Final average overlap: " << sum(overlaps)/num_truncations << endl;
	std::cerr << "Average error corrected overlap: " << sum(overlaps)/num_truncations << endl;

	ofstream out_file(out_file_name);
	out_file << "#NUM_EXPECTED_WALKERS:\n" << num_desired_walkers;
	out_file << "\n#NUM_TRUNCATIONS:\n" << num_truncations;
	out_file << "\n#ORIGINAL_OVERLAP:\n" << original_overlap;
	out_file << "\n#TRUNCATED_OVERLAPS:";
	for(double overlap : overlaps){
		out_file << "\n" << overlap;
	}
	out_file << "\n#NUM_WALKERS:";
	for(int nw : num_walkers){
		out_file << "\n" << nw;
	}
	out_file << "\n#INDIVIDUAL_OVERLAPS:";
	for(vector<double> iovs : individual_overlaps){
		out_file << "\n";
		for(double iov : iovs){
			out_file << iov << " ";
		}
	}
	out_file << "\n#WALKER_WEIGHTS:";
	for(vector<double> wws : walker_weights){
		out_file << "\n";
		for(double ww : wws){
			out_file << ww << " ";
		}
	}
	out_file << "\n#MATRIX_ELEMENTS:";
	for(vector<double> mes : individual_mes){
		out_file << "\n";
		for(double me : mes){
			out_file << me << " ";
		}
	}
	out_file << "\n#TRIAL_OVERLAPS:";
	for(vector<double> fids : individual_fidelities){
		out_file << "\n";
		for(double fid : fids){
			out_file << fid << " ";
		}
	}
	out_file.close();

}