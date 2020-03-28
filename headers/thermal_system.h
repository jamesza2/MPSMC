#ifndef evolving_system
#define evolving_system

#include "itensor/all.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>


//Stores a wavefunction and iterates on it by repeatedly applying itev to it 
class ThermalSystem{
	public:
		itensor::MPS psi;
		itensor::MPO *itev;
		double tau; //Iterate it N times to time evolve by beta = N*tau
		int max_bd;
		int truncated_bd;
		std::mt19937 generator;
		std::uniform_real_distribution<double> distribution;

		ThermalSystem(itensor::SiteSet &sites, 
			itensor::MPO &itev_input, 
			double tau_input, 
			int max_bond_dimension_input, 
			int truncated_bond_dimension_input)
		{
			psi = itensor::randomMPS(sites);
			itev = &itev_input;
			tau = tau_input;
			max_bd = max_bond_dimension_input;
			truncated_bd = truncated_bond_dimension_input;
			distribution = std::uniform_real_distribution<double>(0.0, 1.0);
		}

		double expectation_value(itensor::MPO &A){
			return std::real(itensor::innerC(psi, A, psi))/std::abs(itensor::innerC(psi, psi));
		}

		double overlap(itensor::MPS &psi_other){
			double norm1 = std::sqrt(std::abs(itensor::innerC(psi, psi)));
			double norm2 = std::sqrt(std::abs(itensor::innerC(psi_other, psi_other)));
			return std::real(itensor::innerC(psi_other, psi))/(norm1*norm2);
		}

		//Iterate floor(beta/tau) times, automatically truncating when it gets beyond certain bond dimensions
		void iterate(double beta){
			for(double current_beta = tau; current_beta <= beta; current_beta += tau){
				apply_MPO_no_truncation();
				if(itensor::maxLinkDim(psi) > max_bd){
					truncate();
				}
				double first_tensor_norm = itensor::norm(psi(1));
				psi.ref(1) /= first_tensor_norm;

				double norm = std::sqrt(std::abs(itensor::innerC(psi, psi)));
				if(norm == 0){
					std::cerr << "ERROR: Norm of MPS has dropped to 0" << std::endl;
					print(psi);
					return;
				}
				psi /= norm;
				//double energy = std::real(itensor::innerC(psi, H, psi))/std::abs(itensor::innerC(psi, psi));
				//energy = energy/num_sites;
				//std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has energy " << energy << std::endl;
				//int bond_dimension = itensor::maxLinkDim(psi);
				//double avg_Sz_val = std::real(itensor::innerC(psi, avg_Sz, psi))/std::abs(itensor::innerC(psi, psi));
				
				//out_file << "\n" << energy << "|" << bond_dimension << "|" << avg_Sz_val;
			}
		}

		void iterate_simple(double threshhold){
			apply_MPO_no_truncation();
			if(itensor::maxLinkDim(psi) > max_bd){
				truncate_simple(threshhold);
			}
			double first_tensor_norm = itensor::norm(psi(1));
			psi.ref(1) /= first_tensor_norm;

			double norm = std::sqrt(std::abs(itensor::innerC(psi, psi)));
			if(norm == 0){
				std::cerr << "ERROR: Norm of MPS has dropped to 0" << std::endl;
				print(psi);
				return;
			}
			psi /= norm;
		}

		void iterate_single(){
			iterate(tau);
		}

		void iterate_single_no_truncation(){
			apply_MPO_no_truncation();
		}

		int get_max_bd(){
			return itensor::maxLinkDim(psi);
		}

		int get_avg_bd(){
			return itensor::averageLinkDim(psi);
		}

		void set_truncated_bd(int new_truncated_bd){
			truncated_bd = new_truncated_bd;
		}

		void truncate(){
			//std::cerr << "Truncating MPS..." << std::endl;
			int num_sites = itensor::length(psi);
			//std::mt19937 generator;
			//std::uniform_real_distribution<double> distribution(0.0, 1.0);

			for(int i = 1; i < num_sites; i++){
				auto site_tensor = psi(i);
				auto neighbor_tensor = psi(i+1);
				auto combined_tensor = site_tensor*neighbor_tensor;
				auto left_site = itensor::siteIndex(psi, i);
				itensor::IndexSet Uindices = {left_site};
				if(i > 1){
					auto left_link = itensor::leftLinkIndex(psi, i);
					Uindices = itensor::unionInds(left_link, Uindices);
				}
				auto [U,S,V] = itensor::svd(combined_tensor, Uindices);
				auto V_original_index = itensor::commonIndex(S,V);
				auto U_original_index = itensor::commonIndex(U,S);
				std::vector<double> singular_values = abs_diagonal_elems(S);
				int original_bd = singular_values.size();
				std::vector<int> random_elements = random_weighted(singular_values, truncated_bd, generator, distribution);
				std::vector<std::pair<int, int>> repeats = collect_repeats(random_elements);

				//Turn those random elements into screening matrices to apply to U, S and V
				int final_truncated_bd = repeats.size();
				itensor::Index T_truncated_index(final_truncated_bd,"Link,l="+std::to_string(i));
				itensor::Index T_original_index(original_bd,"original");
				itensor::Index T_truncated_index_primed = itensor::prime(T_truncated_index, 1);
				itensor::ITensor T(T_truncated_index, T_original_index);
				for(int repeat_index = 1; repeat_index <= final_truncated_bd; repeat_index ++){
					T.set(T_truncated_index = repeat_index, T_original_index = repeats[repeat_index-1].first, 1.0);
				}
				
				//Apply them to U, S and V
				//Should change U to U*T, S to T*S*T and V to T*S
				V = V*(T*itensor::delta(T_original_index, V_original_index)*itensor::delta(T_truncated_index, T_truncated_index_primed));
				U = U*(T*itensor::delta(T_original_index, U_original_index));
				S = S*(T*itensor::delta(T_original_index, U_original_index))*(T*itensor::delta(T_original_index,V_original_index)*itensor::delta(T_truncated_index, T_truncated_index_primed));
				//Turn S's diagonal elements into repeat numbers
				for(int repeat_index = 1; repeat_index <= final_truncated_bd; repeat_index ++){
					S.set(T_truncated_index = repeat_index, T_truncated_index_primed = repeat_index, 1.0*repeats[repeat_index-1].second/truncated_bd);
				}
				//Collect new U into current matrix, S*V into the forward matrix
				psi.ref(i) = U;
				psi.ref(i+1) = S*V;
				//Readjust link indices
				auto link_indices = itensor::linkInds(psi);
				link_indices(i) = T_truncated_index;
				psi.replaceLinkInds(link_indices);
			}
		}


		void truncate_simple(double threshhold){ //Truncate by removing lowest singular values (keeping up to max_bd singular values) instead of randomly selecting values
			int num_sites = itensor::length(psi);

			for(int i = 1; i < num_sites; i++){
				auto site_tensor = psi(i);
				auto neighbor_tensor = psi(i+1);
				auto combined_tensor = site_tensor*neighbor_tensor;
				auto left_site = itensor::siteIndex(psi, i);
				itensor::IndexSet Uindices = {left_site};
				if(i > 1){
					auto left_link = itensor::leftLinkIndex(psi, i);
					Uindices = itensor::unionInds(left_link, Uindices);
				}
				auto [U,S,V] = itensor::svd(combined_tensor, Uindices, {"Truncate", true, "MaxDim", max_bd, "Cutoff", threshhold});
				auto V_original_index = itensor::commonIndex(S,V);
				auto U_original_index = itensor::commonIndex(U,S);
				psi.ref(i) = U;
				psi.ref(i+1) = S*V;
				auto link_indices = itensor::linkInds(psi);
				link_indices(i) = U_original_index;
				psi.replaceLinkInds(link_indices);
			}
		}

		itensor::MPS copy_state(){
			return itensor::MPS(psi);
		}

		void set_MPS(itensor::MPS &new_psi){
			psi = itensor::MPS(new_psi);
		}

	private:
		std::vector<double> abs_diagonal_elems(itensor::ITensor &T){
			if(itensor::order(T) != 2){
				std::cerr << "ERROR: Tensor S=";
				Print(T);
				std::cerr << " has order " << itensor::order(T) << " instead of 2" << std::endl;
				return vector<double>();
			}
			auto indices = itensor::inds(T);
			auto index_1 = indices(1);
			auto index_2 = indices(2);
			if(itensor::dim(index_1) != itensor::dim(index_2)){
				std::cerr << "ERROR: Index dimensions " << itensor::dim(index_1) << " and " << itensor::dim(index_2) << " not equal " << std::endl;
				return vector<double>();
			}
			std::vector<double> diagonals;
			diagonals.reserve(itensor::dim(index_1));
			for(int i = 1; i <= itensor::dim(index_1); i++){
				diagonals.push_back(std::abs(itensor::elt(T,index_1=i, index_2=i)));
			}
			return diagonals;
		}

		double sum(std::vector<double> v){
			double s = 0;
			for(double elem:v){
				s += elem;
			}
			return s;
		}

		double random_double(){
			return distribution(generator);
		}

		//Picks a random integer between 0 and len(weights), weighted by the weights std::vector.
		std::vector<int> random_weighted(std::vector<double> weights, int num_picks, std::mt19937 &generator, std::uniform_real_distribution<double> &distribution){
			std::vector<double> cumulative_weights;
			cumulative_weights.push_back(weights[0]);
			for(int i = 1; i < weights.size(); i++){
				cumulative_weights.push_back(weights[i] + cumulative_weights[i-1]);
			}
			double total_weight = cumulative_weights[cumulative_weights.size()-1];
			std::vector<double> rd;
			for(int i = 0; i < num_picks; i++){
				rd.push_back(random_double()*total_weight);
			}
			std::vector<int> random_elements;
			for(double spork:rd){
				int L = 0;
				int R = cumulative_weights.size()-1;
				while(true){
					if(L == R){
						random_elements.push_back(L+1);
						break;
					}
					if(L > R){
						std::cerr << "ERROR: Binary search failed because Left end = " << L << " while Right end = " << R << std::endl;
						random_elements.push_back(-1);
						break;
					}
					int M = std::rint((L+R)/2);
					if(spork > cumulative_weights[M]){
						L = M+1;
						continue;
					}
					if(M == 0){
						random_elements.push_back(1);
						break;
					}
					if(spork < cumulative_weights[M-1]){
						R = M-1;
						continue;
					}
					random_elements.push_back(M+1);
					break;
				}
			}
			return random_elements;

		}

		//Collect repeated instances of n in the vector of integers and count how many times they've been repeated.
		std::vector<std::pair<int, int>> collect_repeats(std::vector<int> random_integers){
			std::vector<std::pair<int, int>> repeats;
			for(int i:random_integers){
				bool repeated = false;
				for(int j = 0; j < repeats.size(); j++){
					std::pair<int, int> j_pair = repeats[j];
					if(i == j_pair.first){
						repeats[j].second += 1;
						repeated = true;
						break;
					}
				}
				if(!repeated){
					repeats.push_back(std::make_pair(i, 1));
				}
			}
			return repeats;
		}

		

		//Apply the MPO to psi, changing it and not attempting to use any algorithms to truncate its bond dimension
		//Automatically deprimes site indices
		void apply_MPO_no_truncation(){
			int num_sites = itensor::length(psi);

			std::vector<itensor::Index> new_link_indices;
			new_link_indices.reserve(num_sites);

			auto MPO_first_link = itensor::rightLinkIndex(*itev, 1);
			auto MPS_first_link = itensor::rightLinkIndex(psi, 1);
			auto [first_combiner, first_link_index] = itensor::combiner(itensor::IndexSet(MPO_first_link, MPS_first_link),{"Tags=","Link,l=1"});

			std::vector<itensor::ITensor> new_MPS;
			
			new_MPS.push_back(psi(1)*(*itev)(1)*first_combiner);
			new_link_indices.push_back(first_link_index);

			for(int i = 2; i <= num_sites; i++){
				auto MPO_left_link = itensor::leftLinkIndex(*itev, i);
				auto MPS_left_link = itensor::leftLinkIndex(psi, i);
				auto [left_combiner, left_combined_index] = itensor::combiner(MPO_left_link, MPS_left_link);
				if(i == num_sites){
					new_MPS.push_back(psi(num_sites)*(*itev)(num_sites)*left_combiner*itensor::delta(left_combined_index, new_link_indices[num_sites-2]));
					break;
				}
				auto MPO_right_link = itensor::rightLinkIndex(*itev, i);
				auto MPS_right_link = itensor::rightLinkIndex(psi, i);
				auto [right_combiner, right_combined_index] = itensor::combiner(itensor::IndexSet(MPO_right_link, MPS_right_link), {"Tags=","Link,l="+std::to_string(i)});
				new_MPS.push_back(psi(i)*(*itev)(i)*left_combiner*right_combiner*itensor::delta(left_combined_index, new_link_indices[i-2]));
				new_link_indices.push_back(right_combined_index);
			}
			for(int i = 1; i <= num_sites; i++){
				psi.ref(i) = new_MPS[i-1];
			}
			psi.replaceLinkInds(itensor::IndexSet(new_link_indices));
			psi.replaceSiteInds(itensor::noPrime(itensor::siteInds(psi)));
		}
};

#endif