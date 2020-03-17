#ifndef operator_maker
#define operator_maker

#include <random>
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
};



#endif