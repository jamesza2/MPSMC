#ifndef operator_maker
#define operator_maker

#include <random>
#include <regex>
#include "itensor/all.h"

class OperatorMaker{
	public:
		OperatorMaker(itensor::SiteSet &sites_input){
			sites = &sites_input;
			num_sites=  itensor::length(*sites);
		}

		itensor::AutoMPO XXZHamiltonian(double Jz, double h){
			auto ampo = itensor::AutoMPO(*sites);
			for(int i = 1; i < num_sites; i++){
				ampo += 0.5,"S+",i,"S-",i+1;
				ampo += 0.5,"S-",i,"S+",i+1;
				ampo += Jz,"Sz",i,"Sz",i+1;
				ampo += h,"Sz",i;
			}
			ampo += h,"Sz", num_sites;
			return ampo;
		}

		itensor::AutoMPO J1J2Hamiltonian(double J2){
			auto ampo = itensor::AutoMPO(*sites);
			for(int i = 1; i < num_sites; i++){
				ampo += 0.5,"S+",i,"S-",i+1;
				ampo += 0.5,"S-",i,"S+",i+1;
				ampo += 1.0,"Sz",i,"Sz",i+1;
			}
			for(int i = 1; i < num_sites-1; i++){
				ampo += 0.5*J2,"S+",i,"S-",i+2;
				ampo += 0.5*J2,"S-",i,"S+",i+2;
				ampo += 1.0*J2,"Sz",i,"Sz",i+2;
			}
			return ampo;
		}

		itensor::AutoMPO J1J3Hamiltonian(double J2, double J3){
			auto ampo = itensor::AutoMPO(*sites);
			for(int i = 1; i < num_sites; i++){
				ampo += 0.5,"S+",i,"S-",i+1;
				ampo += 0.5,"S-",i,"S+",i+1;
				ampo += 1.0,"Sz",i,"Sz",i+1;
			}
			for(int i = 1; i < num_sites-1; i++){
				ampo += 0.5*J2,"S+",i,"S-",i+2;
				ampo += 0.5*J2,"S-",i,"S+",i+2;
				ampo += 1.0*J2,"Sz",i,"Sz",i+2;
			}
			for(int i = 1; i < num_sites-2; i++){
				ampo += 0.5*J3,"S+",i,"S-",i+2;
				ampo += 0.5*J3,"S-",i,"S+",i+2;
				ampo += 1.0*J3,"Sz",i,"Sz",i+2;
			}
			return ampo;
		}

		itensor::AutoMPO Lattice(double J2, std::string bond_list_file_name, int num_sites, int num_sites_per_rung){

			auto ampo = itensor::AutoMPO(*sites);
			std::cout << "Reading bond list file name " << bond_list_file_name << std::endl;
			auto bond_matrices = read_bonds(bond_list_file_name, num_sites, num_sites_per_rung);
			std::vector<std::vector<int>> bond_matrix = bond_matrices[0];
			std::vector<std::vector<int>> nnn_bond_matrix = bond_matrices[1];
			for(int i = 0; i < num_sites; i++){
				for(int j : bond_matrix[i]){
					ampo += 0.5,"S+",i,"S-",j;
					ampo += 0.5,"S-",i,"S+",j;
					ampo += 1.0,"Sz",i,"Sz",j;
					std::cout << "Creating bond at " << i << ", " << j << std::endl;
				}
				
				for(int j : bond_matrix[i]){
					ampo += 0.5*J2,"S+",i,"S-",j;
					ampo += 0.5*J2,"S-",i,"S+",j;
					ampo += 1.0*J2,"Sz",i,"Sz",j;
					std::cout << "Creating NNN bond at " << i << ", " << j << std::endl;
				}

			}
			return ampo;
			
		}

		itensor::AutoMPO AverageSz(){
			auto ampo = itensor::AutoMPO(*sites);
			double factor = 1.0/num_sites;
			for(int i = 1; i <= num_sites; i++){
				ampo += factor,"Sz",i;
			}
			return ampo;
		}

		itensor::AutoMPO RandomFieldsHamiltonian(double W){
			std::mt19937 generator;
			std::uniform_real_distribution<double> distribution(-W, W);
			auto ampo = itensor::AutoMPO(*sites);
			for(int i = 1; i < num_sites; i++){
				ampo += 0.5,"S+",i,"S-",i+1;
				ampo += 0.5,"S-",i,"S+",i+1;
				ampo += 1.0,"Sz",i,"Sz",i+1;
			}
			for(int i = 1; i <= num_sites; i++){
				ampo += distribution(generator),"Sz",i;
			}
			return ampo;
		}

		itensor::AutoMPO SpinCorrelation(int i, int j){
			auto ampo = itensor::AutoMPO(*sites);
			ampo += 0.5,"S+",i,"S-",j;
			ampo += 0.5,"S-",i,"S+",j;
			ampo += 1.0,"Sz",i,"Sz",j;
			return ampo;
		}

		itensor::AutoMPO SingleSiteSz(int i){
			auto ampo = itensor::AutoMPO(*sites);
			ampo += 1.0,"Sz",i;
			return ampo;
		}

		itensor::AutoMPO MPOFromString(std::string key, double args[]){
			auto total_size = sizeof(args);
			size_t num_args = 0;
			if(total_size != 0){
				num_args = total_size/sizeof(args[0]);
			}
			if((key == "XXZ")||(key == "Heisenberg")){
				if(num_args < 2){
					std::cerr << "Error: requires " << 2 << " arguments but found " << num_args << " instead." << std::endl;
					return itensor::AutoMPO(*sites);
				}
				return XXZHamiltonian(args[0], args[1]);
			}
			if((key == "RandomField")||(key == "MBLHamiltonian")){
				if(num_args < 1){
					std::cerr << "Error: requires " << 1 << " arguments but found " << num_args << " instead." << std::endl;
					return itensor::AutoMPO(*sites);
				}
				return RandomFieldsHamiltonian(args[0]);
			}
			if((key == "AverageSz")||(key == "AvgSz")){
				return AverageSz();
			}
		}

	private:
		itensor::SiteSet *sites;
		int num_sites;

		std::vector<std::vector<std::vector<int>>> read_bonds(std::string bond_list_file_name, int num_sites, int num_sites_per_rung){
			std::cerr << std::endl;
			std::vector<std::vector<int>> bond_matrix;
			std::vector<std::vector<int>> nnn_bond_matrix;
			bond_matrix.reserve(num_sites);
			nnn_bond_matrix.reserve(num_sites);
			for(int i = 0; i < num_sites; i++){
				std::vector<int> k;
				bond_matrix.push_back(k);
				std::vector<int> l;
				nnn_bond_matrix.push_back(l);
			}
			std::ifstream blf(bond_list_file_name);
			bool nnn_bond = false;
			std::string line;
			while(std::getline(blf, line)){
				if(line.length() == 0){continue;}
				if(line[0] == '#'){
					if(line == "#NN"){
						nnn_bond = false;
					}else{
						if(line == "#NNN"){
							nnn_bond = true;
						}
						else{
							std::cerr << "ERROR: Cannot parse line " << line << std::endl;
						}
					}
				}
				else{
					std::regex line_reader("(\\d+\\+?) (\\d+\\+?)");
					std::smatch line_parts;
					std::regex_search(line, line_parts, line_reader);
					auto first_index = line_parts[1].str();
					auto second_index = line_parts[2].str();
					//cout << first_index << ";" << second_index << endl;
					int i, j;
					if(first_index.find('+') != std::string::npos){ 
						//i = atoi(first_index.substr(0,first_index.length()-1).c_str()) + num_sites_per_rung;
						i = std::atoi(first_index.c_str()) + num_sites_per_rung;
					}
					else{
						i = std::atoi(first_index.c_str());
					}
					if(second_index.find('+') != std::string::npos){
						//j = atoi(second_index.substr(0,second_index.length()-1).c_str()) + num_sites_per_rung;
						j = std::atoi(second_index.c_str()) + num_sites_per_rung;
					}
					else{
						j = std::atoi(second_index.c_str());
					}
					std::cout << "Adding basic bond " << i << ", " << j << std::endl;
					for(int rung = 0; rung < num_sites; rung+= num_sites_per_rung){
						//cout << "Adding interaction at " << i+rung << ", " << j+rung << endl;
						if((i + rung >= num_sites)||(j+rung >= num_sites)){continue;}
						if(nnn_bond){
							nnn_bond_matrix[i+rung].push_back(j+rung);
						}
						else{
							bond_matrix[i+rung].push_back(j+rung);
						}
					}
				}
			}
			blf.close();
			std::vector<std::vector<std::vector<int>>> bond_matrices;
			bond_matrices.push_back(bond_matrix);
			bond_matrices.push_back(nnn_bond_matrix);
			return bond_matrices;
		}
};



#endif