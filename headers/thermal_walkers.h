#ifndef thermal_walkers
#define thermal_walkers

#include "itensor/all.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>

//Normal: Uses the previous 2 total weights to estimate energy, then attempts to make the total walker weight num_walkers
//Constant: Uses a single estimate for the energy and nothing else
//Antitrunc: Just tries to compensate for the truncation


//Stores a wavefunction and iterates on it by repeatedly applying itev to it 
class ThermalWalkers{
	public:
		vector<itensor::MPS> walkers;
		vector<double> weights;
		itensor::MPO *itev;
		itensor::MPS trial_wavefunction;
		itensor::MPS fixed_node_wavefunction;
		double tau; //Iterate it N times to time evolve by beta = N*tau
		int max_bd;
		int truncated_bd;
		double estimated_error;
		double trial_energy;
		int num_walkers;
		int num_max_walkers;
		std::mt19937 generator;
		std::uniform_real_distribution<double> distribution;
		int kept_singular_values;
		bool verbose;
		bool fixed_node; //If true, extincts any walker whose overlap with the trial wavefunction flips sign
		bool original_overlap_positive;
		enum trial_energy_calculation_mode{NORMAL, CONSTANT, ANTITRUNC, ONLY_ENERGY} te_mode;


		ThermalWalkers(itensor::SiteSet &sites, 
			itensor::MPO &itev_input, 
			double tau_input, 
			int max_bond_dimension_input, 
			int truncated_bond_dimension_input,
			int num_walkers_input,
			int num_max_walkers_input)
		{
			auto psi = itensor::randomMPS(sites);
			psi /= std::sqrt(itensor::inner(psi, psi));
			walkers.clear();
			walkers.push_back(itensor::MPS(psi));
			weights.clear();
			weights.push_back(std::sqrt(std::abs(itensor::innerC(psi, psi))));
			trial_wavefunction = itensor::MPS(psi);
			fixed_node_wavefunction = itensor::MPS(trial_wavefunction);
			itev = &itev_input;
			tau = tau_input;
			max_bd = max_bond_dimension_input;
			truncated_bd = truncated_bond_dimension_input;
			estimated_error = 1.0;
			distribution = std::uniform_real_distribution<double>(0.0, 1.0);
			trial_energy = 0;
			num_walkers = num_walkers_input;
			num_max_walkers = num_max_walkers_input;
			kept_singular_values = 0;
			verbose = false;
			fixed_node = false;
			original_overlap_positive = (itensor::inner(psi, trial_wavefunction) > 0);
			te_mode = NORMAL;
		}

		void set_trial_energy_calculation_mode(std::string te_mode_string){
			if(te_mode_string == "CONSTANT"){
				te_mode = CONSTANT;
			}
			else{
				if(te_mode_string == "ANTITRUNC"){
					te_mode = ANTITRUNC;
				}
				else{
					if(te_mode_string == "ONLY_ENERGY"){
						te_mode = ONLY_ENERGY;
					}
					else{
						if(te_mode_string != "NORMAL"){
							std::cout << "Warning: Couldn't read trial energy calculation mode" << std::endl;
						}
						te_mode = NORMAL;
					}
					
				}
			}
		}

		void set_trial_wavefunction(itensor::MPS &new_trial_wavefunction){
			trial_wavefunction = itensor::MPS(new_trial_wavefunction);
		}

		void set_fixed_node_wavefunction(itensor::MPS &new_fixed_node_wavefunction){
			fixed_node_wavefunction = itensor::MPS(new_fixed_node_wavefunction);
		}

		void set_kept_singular_values(int kept_singular_values_input){
			kept_singular_values = kept_singular_values_input;
		}

		vector<double> expectation_values(itensor::MPO &A){
			//auto trial_wavefunction = walkers[0];
			vector<double> evs;
			int num_sites = itensor::length(walkers[0]);
			for(auto MPS_iter = walkers.begin(); MPS_iter != walkers.end(); ++MPS_iter){

				std::complex<double> me_complex = itensor::innerC(trial_wavefunction, A, *MPS_iter);
				if(std::abs(std::real(me_complex)) < 10.0*std::abs(std::imag(me_complex))){
					std::cerr << "WARNING: Matrix Element of " << me_complex << " has significant imaginary part" << std::endl;
				}
				evs.push_back(std::real(me_complex)/num_sites);
			}
			return evs;
		}



		vector<double> get_weights(){
			vector<double> w;
			for(double weight : weights){
				w.push_back(weight);
			}
			return w;
		}

		vector<int> get_bds(){
			vector<int> bds;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				bds.push_back(itensor::maxLinkDim(walkers[MPS_index]));
			}
			return bds;
		}

		//Unweighted overlaps
		vector<double> overlaps(itensor::MPS &psi_other){
			vector<double> ovs;
			//double norm2 = std::sqrt(std::abs(itensor::innerC(psi_other, psi_other)));
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				//double norm1 = std::sqrt(std::abs(itensor::innerC(*MPS_iter, *MPS_iter)));
				std::complex<double> overlap_complex = itensor::innerC(walkers[MPS_index], psi_other);
				if(std::abs(std::real(overlap_complex)) < 10.0*std::abs(std::imag(overlap_complex))){
					std::cerr << "WARNING: Overlap of " << overlap_complex << " has significant imaginary part" << std::endl;
				}
				ovs.push_back(std::real(overlap_complex)/weights[MPS_index]);
			}
			return ovs;
		}

		vector<double> weighted_overlaps(itensor::MPS &psi_other){
			vector<double> ovs;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				std::complex<double> overlap_complex = itensor::innerC(walkers[MPS_index], psi_other);
				if(std::abs(std::real(overlap_complex)) < 10.0*std::abs(std::imag(overlap_complex))){
					std::cerr << "WARNING: Overlap of " << overlap_complex << " has significant imaginary part" << std::endl;
				}
				ovs.push_back(std::real(overlap_complex));
			}
			return ovs;
		}

		//Overlap <psi_other|psi> weighted by weights (no need to multiply weights as we already have |psi>)
		double overlap(itensor::MPS &psi_other){
			double ov = 0;
			for(auto MPS_iter = walkers.begin(); MPS_iter != walkers.end(); ++MPS_iter){
				ov += std::real(itensor::innerC(*MPS_iter, psi_other));
			}
			return ov;
		}

		//Energies of each MPS weighted by weights
		double expectation_value(itensor::MPO &A){
			//auto trial_wavefunction = walkers[0];
			double weighted_energy = 0;
			double weighted_sum = 0;

			int num_sites = itensor::length(walkers[0]);
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				weighted_energy += std::real(itensor::innerC(trial_wavefunction, A, walkers[MPS_index]));
				weighted_sum += std::real(itensor::innerC(trial_wavefunction, walkers[MPS_index]));
			}
			return weighted_energy/(weighted_sum*num_sites);
		}

		//Unweighted average energy of each MPS
		double average_walker_energy(itensor::MPO &A){
			double en = 0;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				en += std::real(itensor::innerC(walkers[MPS_index], A, walkers[MPS_index]))/(weights[MPS_index]*weights[MPS_index]);
			}
			return en/walkers.size();
		}

		double average_entanglement_entropy(int site_of_cut){
			double total_ee = 0;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				auto[U,S,V] = svd_on_site(MPS_index, site_of_cut);
				vector<double> singular_values = abs_diagonal_elems(S);
				for(double sv : singular_values){
					double svsq = sv*sv/(weights[MPS_index]*weights[MPS_index]);
					if(svsq != 0){
						total_ee -= svsq*std::log(svsq);
					}
				}
			}
			return total_ee/walkers.size();
		}

		//Iterate floor(beta/tau) times, automatically truncating when it gets beyond certain bond dimensions
		void iterate(double beta){
			for(double current_beta = tau; current_beta <= beta; current_beta += tau){
				double old_weight_sum = norm(weights);
				apply_MPO_no_truncation();
				double new_weight_sum = norm(weights);
				process();
				double after_trunc_weight_sum = norm(weights);
				if(te_mode == NORMAL){
					recalculate_trial_energy(std::log(old_weight_sum/new_weight_sum));
				}
				if(te_mode == ANTITRUNC){
					recalculate_trial_energy(std::log(new_weight_sum/after_trunc_weight_sum)/tau);
				}
				if(te_mode == CONSTANT){
					recalculate_trial_energy(-0.4386);
				}
			}
		}

		void process(){
			/*std::cerr << "  Walker weights: ";
			for(double weight : weights){
				std::cerr << weight << " ";
			}
			std::cerr << std::endl;*/
			combine_walkers();
			split_walkers();
			/*std::cerr << "    Recombined walker weights: ";
			for(double weight : weights){
				std::cerr << weight << " ";
			}
			std::cerr << std::endl;*/
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				if(itensor::maxLinkDim(walkers[MPS_index]) > max_bd){
					itensor::MPS original(walkers[MPS_index]);
					double old_weight = weights[MPS_index];
					if(verbose){
						std::cerr << "Truncating MPS #" << MPS_index << "..." << std::endl;
					}
					
					truncate_single_MPS(MPS_index);
					reweight(MPS_index);
					if(verbose){
						std::cerr << "FIDELITY AFTER TRUNCATION: " << itensor::inner(original, walkers[MPS_index])/(old_weight*weights[MPS_index]) << std::endl;
					}
				}
			}
			//For each walker:
			//If the overlap with the trial wavefunction flips sign, kill it
			//But if all overlaps flip sign, keep them all
			if(fixed_node){
				vector<double> ovs = overlaps(fixed_node_wavefunction);
				bool all_overlaps_flip_sign = true;
				for(double ov : ovs){
					bool overlap_positive = (ov > 0);
					if(overlap_positive == original_overlap_positive){
						all_overlaps_flip_sign = false;
					}
				}
				if(all_overlaps_flip_sign){
					original_overlap_positive = !original_overlap_positive;
					std::cerr << "WARNING: ALL OVERLAPS FLIPPED SIGN" << std::endl;
				}
				else{
					for(int MPS_index = walkers.size()-1; MPS_index >= 0; MPS_index --){
						double ov = ovs[MPS_index];
						bool overlap_positive = (ov > 0);
						if(overlap_positive != original_overlap_positive){
							walkers.erase(walkers.begin() + MPS_index);
							weights.erase(weights.begin() + MPS_index);
							std::cerr << "  ERASING WALKER #" << MPS_index << " DUE TO OVERLAP " << ov << " BEING THE WRONG SIGN" << std::endl;
						}
					}
				}

			}
			/*std::cerr << "      New walker weights: ";
			for(double weight : weights){
				std::cerr << weight << " ";
			}
			std::cerr << std::endl;*/
			//recalculate_trial_energy();
		}

		void reweight(int MPS_index, double new_weight = -1){
			itensor::MPS & psi = walkers.at(MPS_index);
			double norm = std::sqrt(std::abs(itensor::innerC(psi, psi)));
			if(new_weight != -1){
				psi /= (norm/new_weight);
				weights[MPS_index] = new_weight;
			}
			else{
				weights[MPS_index] = norm;
			}
		}


		void combine_walkers(){
			vector<int> to_combine;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				if(weights[MPS_index] < 0.5){
					to_combine.push_back(MPS_index);
				}
			}
			int index_to_combine = -1;
			vector<int> to_remove;
			for(int MPS_index : to_combine){
				if(index_to_combine > -1){
					double first_weight = weights[index_to_combine];
					double second_weight = weights[MPS_index];
					double selector = random_double()*(first_weight + second_weight);
					int chosen_index = MPS_index;
					if(selector <= first_weight){
						reweight(index_to_combine, first_weight + second_weight);
						to_remove.insert(to_remove.begin(), MPS_index);
						chosen_index = index_to_combine;
					}
					else{
						reweight(MPS_index, first_weight + second_weight);
						to_remove.insert(to_remove.begin(), index_to_combine);
					}
					//std::cerr << "Combining walkers #" << index_to_combine << "(weight " << first_weight << ") and #" << MPS_index << "(weight " << second_weight << "), choosing " << chosen_index << std::endl;
					index_to_combine = -1;
				}
				else{
					index_to_combine = MPS_index;
				}
			}
			for(int remove_index : to_remove){
				walkers.erase(walkers.begin() + remove_index);
				weights.erase(weights.begin() + remove_index);
			}
		}

		void split_walkers(){
			vector<int> to_split;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				if(weights[MPS_index] > 2){
					/*if(to_split.size() < 1){
						to_split.push_back(MPS_index);
					}*/
					bool stick_at_end = true;
					for(auto split_iterator = to_split.begin(); split_iterator != to_split.end(); ++split_iterator){
						if(weights[*split_iterator] < weights[MPS_index]){
							stick_at_end = false;
							to_split.insert(split_iterator, MPS_index);
							break;
						}
					}
					if(stick_at_end){
						to_split.push_back(MPS_index);
					}
					//to_split.push_back(MPS_index);
				}
			}
			for(int MPS_index : to_split){
				if(walkers.size() < num_max_walkers){
					double old_weight = weights[MPS_index];
					reweight(MPS_index, old_weight/2);
					walkers.push_back(itensor::MPS(walkers[MPS_index]));
					weights.push_back(old_weight/2);
					//std::cerr << "Split walker #" << MPS_index << " with original weight " << old_weight << std::endl;
				}
			}
		}

		void recalculate_trial_energy(double expected_energy = 0){
			double total_weight = sum(weights);
			double bounded_energy = std::min(std::max(expected_energy, -1.),0.);
			if(te_mode == NORMAL){
				trial_energy = std::log(static_cast<double>(num_walkers)/total_weight)/tau + bounded_energy;
			}
			if((te_mode == CONSTANT) || (te_mode == ANTITRUNC)){
				trial_energy = bounded_energy;
			}
		}

		void iterate_single(){
			iterate(tau);
		}

		void iterate_single_no_truncation(){
			apply_MPO_no_truncation();
		}

		int get_max_bd(){
			int max_bd = 0;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				max_bd = std::max(max_bd,itensor::maxLinkDim(walkers[MPS_index]));
			}
			return max_bd;
		}

		void set_max_bd(int new_max_bd){
			max_bd = new_max_bd;
		}

		void set_truncated_bd(int new_truncated_bd){
			truncated_bd = new_truncated_bd;
		}

		std::tuple<itensor::ITensor, itensor::ITensor, itensor::ITensor> svd_on_site(int MPS_index, int site){
			auto psi = walkers[MPS_index];
			auto site_tensor = psi(site);
			auto neighbor_tensor = psi(site+1);
			auto combined_tensor = site_tensor*neighbor_tensor;
			auto left_site = itensor::siteIndex(psi, site);
			itensor::IndexSet Uindices = {left_site};
			if(site > 1){
				auto left_link = itensor::leftLinkIndex(psi, site);
				Uindices = itensor::unionInds(left_link, Uindices);
			}
			return itensor::svd(combined_tensor, Uindices);
		}

		void truncate_single_site_single_MPS(int site, int MPS_index){
			//std::cerr << "Performing SVD on site..." << std::endl;
			auto[U,S,V] = svd_on_site(MPS_index, site);
			//Print(S);
			auto V_original_index = itensor::commonIndex(S,V);
			auto U_original_index = itensor::commonIndex(U,S);
			std::vector<double> singular_values = abs_diagonal_elems(S);
			if(verbose){
				std::cerr << "  Pre-truncation singular values: ";
				for(double sv : singular_values){
					std::cerr << sv << " ";
				}
				std::cerr << std::endl;
			}
			
			int original_bd = singular_values.size();
			std::vector<int> repeats(original_bd,0);
			vector<int> truncated_repeats;
			vector<int> original_indices;
			if(kept_singular_values < original_bd){
				for(int excluded_index = 0; excluded_index < kept_singular_values; excluded_index++){
					singular_values[excluded_index] = 0;
				}
				int final_truncated_bd = random_weighted(singular_values, truncated_bd, repeats);
				
				for(int i = 0; i < repeats.size(); i++){
					if(repeats[i] != 0){
						truncated_repeats.push_back(repeats[i]);
						original_indices.push_back(i+1);
					}
				}
				if(verbose){
					std::cerr << "   Randomly selected indices: ";
					for(int rindex = 0; rindex < original_indices.size(); rindex ++){
						std::cerr << original_indices[rindex] << "(" << truncated_repeats[rindex] << " times)| ";
					}
					std::cerr << std::endl;
				}
				
			}
			double singular_value_weight = 1;
			if(truncated_bd != 0){
				singular_value_weight = sum(singular_values)/truncated_bd;
			}
			if(verbose){
				std::cerr << "   New set weight: " << singular_value_weight << std::endl;
			}
			//std::cerr << "Truncating after selecting singular values..." << std::endl;
			truncate_based_on_selection(truncated_repeats, original_indices, U, S, V, original_bd, site, MPS_index, singular_value_weight);
		}
		void truncate_single_MPS(int MPS_index){
			int num_sites = itensor::length(walkers[MPS_index]);
			for(int site = 1; site < num_sites; site++){
				if(verbose){
					std::cerr << " On site " << site << "..." << std::endl;
				}
				truncate_single_site_single_MPS(site, MPS_index);
			}
		}

		void truncate(){
			//int num_sites = itensor::length(walkers[0]);
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				if(itensor::maxLinkDim(walkers[MPS_index]) > max_bd){
					truncate_single_MPS(MPS_index);
				}
			}
		}

		void truncate_based_on_selection(std::vector<int> &truncated_repeats, 
				std::vector<int> &original_indices,
				itensor::ITensor &U,
				itensor::ITensor &S,
				itensor::ITensor &V,
				int original_bd,
				int site,
				int MPS_index,
				double singular_value_weight)
		{
			std::vector<double> singular_values = abs_diagonal_elems(S);
			auto V_original_index = itensor::commonIndex(S,V);
			auto U_original_index = itensor::commonIndex(U,S);
			itensor::MPS & psi = walkers.at(MPS_index);
			if(kept_singular_values >= original_bd){
				//Do nothing to truncate or change S
				psi.ref(site) = U;
				psi.ref(site+1) = S*V;
				auto link_indices = itensor::linkInds(psi);
				link_indices(site) = U_original_index;
				psi.replaceLinkInds(link_indices);
				return;
			}
			
			int final_truncated_bd = truncated_repeats.size();
			//Turn those random elements into screening matrices to apply to U, S and V
			//std::cerr << "Creating truncation tensor...";
			itensor::Index T_truncated_index(final_truncated_bd+kept_singular_values,"Link,l="+std::to_string(site));
			itensor::Index T_original_index(original_bd,"original");
			itensor::Index T_truncated_index_primed = itensor::prime(T_truncated_index, 1);
			itensor::ITensor T(T_truncated_index, T_original_index);
			for(int repeat_index = 1; repeat_index <= kept_singular_values; repeat_index ++){
				T.set(T_truncated_index = repeat_index, T_original_index = repeat_index, 1.0);
			}

			for(int repeat_index = 1; repeat_index <= final_truncated_bd; repeat_index ++){
				T.set(T_truncated_index = repeat_index + kept_singular_values, T_original_index = original_indices[repeat_index-1], 1.0);
			}
			//Print(T);
			
			//Apply them to U, S and V
			//Should change U to U*T, S to T*S*T and V to T*S
			//std::cerr << "Truncating V, U and S..." << std::endl;
			V = V*(T*itensor::delta(T_original_index, V_original_index)*itensor::delta(T_truncated_index, T_truncated_index_primed));
			U = U*(T*itensor::delta(T_original_index, U_original_index));
			S = S*(T*itensor::delta(T_original_index, U_original_index))*(T*itensor::delta(T_original_index,V_original_index)*itensor::delta(T_truncated_index, T_truncated_index_primed));
			//Turn S's diagonal elements into repeat numbers
			for(int repeat_index = 1; repeat_index <= kept_singular_values; repeat_index ++){
				S.set(T_truncated_index = repeat_index, T_truncated_index_primed = repeat_index, singular_values[repeat_index-1]);
			}
			for(int repeat_index = 1; repeat_index <= final_truncated_bd; repeat_index ++){
				S.set(T_truncated_index = repeat_index + kept_singular_values, T_truncated_index_primed = repeat_index + kept_singular_values, singular_value_weight*truncated_repeats[repeat_index-1]);
			}
			std::vector<double> new_singular_values = abs_diagonal_elems(S);
			if(verbose){
				std::cerr << "  Post-truncation singular values: ";
				for(double sv : new_singular_values){
					std::cerr << sv << " ";
				}
				std::cerr << std::endl;
			}

			//Collect new U into current matrix, S*V into the forward matrix
			psi.ref(site) = U;
			psi.ref(site+1) = S*V;
			//Readjust link indices
			auto link_indices = itensor::linkInds(psi);
			link_indices(site) = T_truncated_index;
			psi.replaceLinkInds(link_indices);
		}

		itensor::MPS copy_state(int MPS_index){
			return itensor::MPS(walkers[MPS_index]);
		}

		void set_MPS(itensor::MPS &new_psi, int MPS_index){
			walkers[MPS_index] = itensor::MPS(new_psi);
			reweight(MPS_index);
		}

		void set_all_MPS(vector<itensor::MPS> &new_psis){
			if(new_psis.size() >= walkers.size()){
				for(int MPS_index = 0; MPS_index < new_psis.size(); MPS_index ++){
					if(MPS_index < walkers.size()){
						set_MPS(new_psis[MPS_index], MPS_index);
					}
					else{
						walkers.push_back(itensor::MPS(new_psis[MPS_index]));
						weights.push_back(std::sqrt(std::abs(itensor::innerC(new_psis[MPS_index],new_psis[MPS_index]))));
					}
				}
			}
			else{
				for(int MPS_index = 0; MPS_index < new_psis.size(); MPS_index ++){
					set_MPS(new_psis[MPS_index], MPS_index);
				}
				while(walkers.size() > new_psis.size()){
					walkers.pop_back();
					weights.pop_back();
				}
			}
		}

		vector<itensor::MPS> get_all_MPS(){
			vector<itensor::MPS> all_MPS;
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				all_MPS.push_back(itensor::MPS(walkers[MPS_index]));
			}
			return all_MPS;
		}

		void shave(int final_num_walkers){
			while(walkers.size() > final_num_walkers){
				walkers.pop_back();
				weights.pop_back();
			}
		}

		double random_double(){
			return distribution(generator);
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

		double norm(std::vector<double> v){
			double n = 0;
			for(double elem:v){
				n += elem*elem;
			}
			return std::sqrt(n);
		}

		int norm_squared(std::vector<int> v){
			int ns = 0;
			for(int elem:v){
				ns += elem*elem;
			}
			return ns;
		}

		

		std::vector<double> accumulate_weights(std::vector<double> &weights){
			std::vector<double> cumulative_weights;
			cumulative_weights.push_back(weights[0]);
			for(int i = 1; i < weights.size(); i++){
				cumulative_weights.push_back(weights[i] + cumulative_weights[i-1]);
			}
			return cumulative_weights;
		}

		//Selects a random index i weighted by weights[i]*new_norm/old_norm where the norm is sqrt(sum of repeats^2)
		//Automatically updates the repeats vector and returns 1 if the index was never selected before
		int random_norm_weighted(std::vector<double> &weights, std::vector<int> &repeats){
			int old_norm_squared = 0;
			for(int r : repeats){
				old_norm_squared += r*r;
			}
			double old_norm = std::sqrt(old_norm_squared);

			std::vector<double> cumulative_weights;
			for(int i = 0; i < weights.size(); i++){
				double adjusted_weight = weights[i];
				if(old_norm_squared != 0){
					adjusted_weight *= std::sqrt(old_norm_squared + 2*repeats[i] + 1)/old_norm;
				}
				if(i != 0){
					adjusted_weight += cumulative_weights[i-1];
				}
				cumulative_weights.push_back(adjusted_weight);
			}
			double total_weight = cumulative_weights[cumulative_weights.size()-1];
			double rd = random_double()*total_weight;
			int L = 0;
			int R = cumulative_weights.size()-1;
			int selected_index = 0;
			while(true){
				if(L == R){
					selected_index = L;
					break;
				}
				if(L > R){
					std::cerr << "ERROR: Binary search failed because Left end = " << L << " while Right end = " << R << std::endl;
					selected_index = -1;
					break;
				}
				int M = std::rint((L+R)/2);
				if(rd > cumulative_weights[M]){
					L = M+1;
					continue;
				}
				if(M == 0){
					selected_index = 0;
					break;
				}
				if(rd < cumulative_weights[M-1]){
					R = M-1;
					continue;
				}
				selected_index = M;
				break;
			}
			repeats[selected_index] += 1;
			if(repeats[selected_index] == 1){
				return 1;
			}
			return 0;
		}

		int random_cumulative_weighted_single(std::vector<double> cumulative_weights){
			double total_weight = cumulative_weights[cumulative_weights.size()-1];
			double spork = random_double()*total_weight;
			//std::vector<int> random_elements;
			int num_unique_selections = 0;
			int L = 0;
			int R = cumulative_weights.size()-1;
			int random_element = 0;
			while(true){
				if(L == R){
					random_element = L;
					//random_elements.push_back(L+1);
					break;
				}
				if(L > R){
					std::cerr << "ERROR: Binary search failed because Left end = " << L << " while Right end = " << R << std::endl;
					random_element = -1;
					//random_elements.push_back(-1);
					break;
				}
				int M = std::rint((L+R)/2);
				if(spork > cumulative_weights[M]){
					L = M+1;
					continue;
				}
				if(M == 0){
					random_element = 0;
					//random_elements.push_back(1);
					break;
				}
				if(spork < cumulative_weights[M-1]){
					R = M-1;
					continue;
				}
				random_element = M;
				//random_elements.push_back(M+1);
				break;
			}
			return random_element;
		}

		int random_weighted_single(std::vector<double> &weights){
			std::vector<double> cumulative_weights = accumulate_weights(weights);
			return random_cumulative_weighted_single(cumulative_weights);
		}

		//Picks a random integer between 0 and len(weights), weighted by the weights std::vector.
		//Stores data in the repeats vector and returns the number of unique selections
		int random_weighted(std::vector<double> &weights, int num_picks, std::vector<int> &repeats){
			std::vector<double> cumulative_weights = accumulate_weights(weights);
			int num_unique_selections = 0;
			for(int i = 0; i < num_picks; i++){
				int random_element = random_cumulative_weighted_single(cumulative_weights);
				if(repeats[random_element] == 0){
					num_unique_selections += 1;
				}
				repeats[random_element] += 1;
			}
			return num_unique_selections;
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
			for(int MPS_index = 0; MPS_index < walkers.size(); MPS_index ++){
				itensor::MPS old_psi(walkers[MPS_index]);
				double old_weight = weights[MPS_index];
				itensor::MPS & psi = walkers.at(MPS_index);
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
				psi *= std::exp(trial_energy*tau);
				weights[MPS_index] = std::sqrt(std::abs(itensor::innerC(psi, psi)));
				if(verbose){
					std::cerr << "FIDELITY AFTER ITE: " << itensor::inner(old_psi, walkers[MPS_index])/(old_weight*weights[MPS_index]) << std::endl;
				}
			}
			
		}
};

#endif