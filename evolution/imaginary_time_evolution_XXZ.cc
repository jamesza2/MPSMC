#include "itensor/all.h"
#include "../headers/input.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>


std::vector<double> abs_diagonal_elems(itensor::ITensor &T){
	if(itensor::order(T) != 2){
		std::cerr << "ERROR: Tensor S=";
		Print(T);
		std::cerr << " has order " << itensor::order(T) << " instead of 2" << endl;
		return vector<double>();
	}
	auto indices = itensor::inds(T);
	auto index_1 = indices(1);
	auto index_2 = indices(2);
	if(itensor::dim(index_1) != itensor::dim(index_2)){
		std::cerr << "ERROR: Index dimensions " << itensor::dim(index_1) << " and " << itensor::dim(index_2) << " not equal " << endl;
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

double random_double(std::mt19937 &generator, std::uniform_real_distribution<double> &distribution){
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
		rd.push_back(random_double(generator, distribution)*total_weight);
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
				std::cerr << "ERROR: Binary search failed because Left end = " << L << " while Right end = " << R << endl;
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
			pair<int, int> j_pair = repeats[j];
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

void truncate(itensor::MPS &psi, int truncated_bd){
	std::cerr << "Truncating MPS..." << endl;
	int num_sites = itensor::length(psi);
	std::mt19937 generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	for(int i = 2; i <= num_sites; i++){
		//std::cerr << "Site " << i  << ": ";
		auto site_tensor = psi(i);
		//std::cerr << "Original site tensor: " << endl;
		//Print(psi(i));
		auto left_link = itensor::leftLinkIndex(psi, i);
		auto site_index = itensor::siteIndex(psi, i);
		auto [U,S,V] = itensor::svd(site_tensor,{left_link});
		auto V_original_index = itensor::commonIndex(S,V);
		auto U_original_index = itensor::commonIndex(U,S);
		
		//Find the random elements that you're going to keep
		//std::cerr << "Finding random elements... ";
		/*Print(U);
		PrintData(S);
		Print(V);*/
		std::vector<double> singular_values = abs_diagonal_elems(S);
		int original_bd = singular_values.size();
		std::vector<int> random_elements = random_weighted(singular_values, truncated_bd, generator, distribution);
		/*std::cerr << "Selected random indices ";
		for(int elem:random_elements){
			std::cerr << elem << " (original weight: " << singular_values[elem-1] << ");";
		}*/
		//std::cerr << endl;
		std::vector<std::pair<int, int>> repeats = collect_repeats(random_elements);

		//Turn those random elements into screening matrices to apply to U, S and V
		//std::cerr << "Creating masking matrix... ";
		int final_truncated_bd = repeats.size();
		//std::cerr << "truncated index, BD " << final_truncated_bd << "... ";
		itensor::Index T_truncated_index(final_truncated_bd,"Link,l="+to_string(i));
		//std::cerr << "original index, BD " << original_bd << "... ";
		itensor::Index T_original_index(original_bd,"original");
		itensor::Index T_truncated_index_primed = itensor::prime(T_truncated_index, 1);
		//std::cerr << "tensor... ";
		itensor::ITensor T(T_truncated_index, T_original_index);
		for(int repeat_index = 1; repeat_index <= final_truncated_bd; repeat_index ++){
			//std::cerr << " Adding element at " << repeat_index << ", " << repeats[repeat_index-1].first;
			T.set(T_truncated_index = repeat_index, T_original_index = repeats[repeat_index-1].first, 1.0);
		}

		//Apply them to U, S and V
		//Should change U to U*T, S to T*S*T and V to T*S
		//std::cerr << "Masking USV matrices... ";
		V = V*(T*itensor::delta(T_original_index, V_original_index));
		U = U*(T*itensor::delta(T_original_index, U_original_index)*itensor::delta(T_truncated_index, T_truncated_index_primed));
		S = S*(T*itensor::delta(T_original_index, U_original_index)*itensor::delta(T_truncated_index, T_truncated_index_primed))*(T*itensor::delta(T_original_index,V_original_index));
		
		//Print(U);
		//PrintData(S);
		//Print(V);
		//Turn S's diagonal elements into repeat numbers
		//std::cerr << "Readjusting S elements... ";
		for(int repeat_index = 1; repeat_index <= final_truncated_bd; repeat_index ++){
			S.set(T_truncated_index = repeat_index, T_truncated_index_primed = repeat_index, 1.0*repeats[repeat_index-1].second/truncated_bd);
		}
		//Collect new U into current matrix, S*V into the forward matrix
		//std::cerr << "Appling USV matrices to MPS... ";
		/*
		if(i == 2){
			if(itensor::norm(psi(i-1)*U*S) == 0){
				std::cerr << "Warning: Norm of back tensor is 0!" << endl;
				PrintData(U);
				PrintData(S);
				PrintData(U*S);
				PrintData(V);
				PrintData(psi(i-1)*U*S);
			}
		}*/
		psi.ref(i-1) *= U*S;
		psi.ref(i) = V;
		//Readjust link indices
		auto link_indices = itensor::linkInds(psi);
		link_indices(i-1) = T_truncated_index;
		//std::cerr << "Replacing link indices... ";
		psi.replaceLinkInds(link_indices);

		//std::cerr << "Adjusted BD from " << original_bd << " to " << final_truncated_bd << endl;
	}
	
}

//Apply the MPO to psi, changing it and not attempting to use any algorithms to truncate its bond dimension
//Automatically deprimes site indices
void apply_MPO_no_truncation(itensor::MPO &A, itensor::MPS &psi){
	int num_sites = itensor::length(psi);

	std::vector<itensor::Index> new_link_indices;
	new_link_indices.reserve(num_sites);

	auto MPO_first_link = itensor::rightLinkIndex(A, 1);
	auto MPS_first_link = itensor::rightLinkIndex(psi, 1);
	auto [first_combiner, first_link_index] = itensor::combiner(itensor::IndexSet(MPO_first_link, MPS_first_link),{"Tags=","Link,l=1"});

	std::vector<itensor::ITensor> new_MPS;
	
	//Print(first_link_index);
	//first_link_index.replaceTags("CMB","l=1");
	//Print(first_link_index);
	//Print(first_combiner);
	new_MPS.push_back(psi(1)*A(1)*first_combiner);
	new_link_indices.push_back(first_link_index);

	for(int i = 2; i <= num_sites; i++){
		auto MPO_left_link = itensor::leftLinkIndex(A, i);
		auto MPS_left_link = itensor::leftLinkIndex(psi, i);
		auto [left_combiner, left_combined_index] = itensor::combiner(MPO_left_link, MPS_left_link);
		if(i == num_sites){
			new_MPS.push_back(psi(num_sites)*A(num_sites)*left_combiner*itensor::delta(left_combined_index, new_link_indices[num_sites-2]));
			//psi.ref(num_sites) = psi.ref(num_sites)*A.ref(num_sites)*left_combiner*itensor::delta(left_combined_index, new_link_indices[num_sites-2]);
			break;
		}
		auto MPO_right_link = itensor::rightLinkIndex(A, i);
		auto MPS_right_link = itensor::rightLinkIndex(psi, i);
		auto [right_combiner, right_combined_index] = itensor::combiner(itensor::IndexSet(MPO_right_link, MPS_right_link), {"Tags=","Link,l="+to_string(i)});
		/*Print(psi(i));
		Print(psi(i-1));
		Print(left_combined_index);
		Print(left_combiner);
		Print(new_link_indices[i-2]);*/
		
		//psi.ref(i) = psi.ref(i)*A.ref(i)*left_combiner*right_combiner*itensor::delta(left_combined_index, new_link_indices[i-2]);
		//right_combined_index.replaceTags("CMB","l="+to_string(i));
		//right_combined_index.setTags(itensor::tags(MPS_right_link));
		new_MPS.push_back(psi(i)*A(i)*left_combiner*right_combiner*itensor::delta(left_combined_index, new_link_indices[i-2]));
		new_link_indices.push_back(right_combined_index);
	}
	//cerr << "\nUpdating matrices...";
	for(int i = 1; i <= num_sites; i++){
		psi.ref(i) = new_MPS[i-1];
	}

	//cerr << "Updating link indices...";
	/*Print(itensor::siteInds(psi));
	Print(itensor::linkInds(psi));
	Print(itensor::noPrime(itensor::siteInds(psi)));
	Print(itensor::IndexSet(new_link_indices));*/
	psi.replaceLinkInds(itensor::IndexSet(new_link_indices));

	//cerr << "Depriming site indices...";
	psi.replaceSiteInds(itensor::noPrime(itensor::siteInds(psi)));
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
	std::string mps_file_name = input.GetVariable("mps_file_name");
	std::string out_file_name = input.GetVariable("out_file");

	std::cerr << "Read input files" << endl;

	//Create XXZ Hamiltonian

	itensor::SiteSet sites = itensor::SpinHalf(num_sites, {"ConserveQNs=", false});
	auto ampo = itensor::AutoMPO(sites);
	for(int i = 1; i < num_sites; i++){
		ampo += 0.5,"S+",i,"S-",i+1;
		ampo += 0.5,"S-",i,"S+",i+1;
		ampo += Jz,"Sz",i,"Sz",i+1;
		ampo += h,"Sz",i;
	}
	ampo += h,"Sz", num_sites;

	itensor::MPO itev = itensor::toExpH(ampo, tau);
	itensor::MPO H = itensor::toMPO(ampo);

	std::cerr << "Created H and exp(-tau*H)" << endl;
	//Perform imaginary time evolution

	auto avg_Sz_ampo = itensor::AutoMPO(sites);
	for(int i = 1; i <= num_sites; i++){
		avg_Sz_ampo += 1.0/num_sites,"Sz",i;
	}
	itensor::MPO avg_Sz = itensor::toMPO(avg_Sz_ampo);
	//Print(avg_Sz);
	//Print(H);

	auto psi = itensor::randomMPS(sites);
	std::ofstream out_file(out_file_name);
	out_file << "Energy|Bond Dimension|Max Sz" << endl;
	for(int iteration = 0; iteration < num_iterations; iteration++){
		//Print(psi);
		//std::cerr << "Applying MPO to psi..." << endl;
		//PrintData(psi);
		//PrintData(itev);
		apply_MPO_no_truncation(itev, psi);
		double first_tensor_norm = itensor::norm(psi(1));
		if(first_tensor_norm == 0){
			std::cerr << "Warning: Norm of back tensor is 0!" << endl;
			PrintData(psi(1));
		}
		psi.ref(1) /= first_tensor_norm;
		//std::cerr << "First tensor:" << endl;
		//Print(psi(1));
		
		//std::cerr << "Printing...";
		//Print(psi);
		//Print(itensor::linkInds(psi));
		
		//std::cerr << "Applying H*psi, max bond dimension now " << itensor::maxLinkDim(psi) << endl;
		if(itensor::maxLinkDim(psi) > max_bd){
			truncate(psi, truncated_bd);
		}
		//std::cerr << "First tensor after truncation:" << endl;
		//Print(psi(1));
		first_tensor_norm = itensor::norm(psi(1));
		psi.ref(1) /= first_tensor_norm;
		//std::cerr << "Norm: " << itensor::norm(psi(1)) << endl;
		double norm = std::abs(itensor::innerC(psi, psi));
		if(norm == 0){
			std::cerr << "ERROR: Norm of MPS has dropped to 0" << endl;
			print(psi);
			return 1;
		}
		double energy = std::real(itensor::innerC(psi, H, psi))/std::abs(itensor::innerC(psi, psi));
		energy = energy/num_sites;
		std::cerr << "Iteration " << iteration+1  << "/" << num_iterations << " has energy " << energy << endl;

		//Write properties of the iteration e.g. max BD, energy, <Sz>

		int bond_dimension = itensor::maxLinkDim(psi);
		//std::cerr << "Computing average Sz..." << endl;
		//Print(psi);
		//Print(avgSz);
		double avg_Sz_val = std::real(itensor::innerC(psi, avg_Sz, psi))/std::abs(itensor::innerC(psi, psi));
		
		out_file << energy << "|" << bond_dimension << "|" << avg_Sz_val << endl;
	}
	out_file.close();

	//Save the MPS for future use
	itensor::writeToFile(mps_file_name, psi);
}