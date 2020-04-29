#ifndef random_MPS_general_BD
#define random_MPS_general_BD
#include "itensor/all.h"
#include <random>
#include <ctime>
#include <cmath>
#include <complex>


//correlation_length determines the degree of off-diagonality of each matrix
namespace randomMPS{
	itensor::MPS randomMPS(itensor::SiteSet &sites, int max_bd, double correlation_length){
		std::uniform_real_distribution<double> distribution = std::uniform_real_distribution<double>(0.0, 1.0);
		std::mt19937 generator;
		int num_sites = itensor::length(sites);
		vector<itensor::Index> link_indices();
		auto psi = itensor::MPS(sites);
		int left_link_dim = max_bd;
		int right_link_dim = max_bd;

		//Leftmost tensor
		itensor::Index first_right_link(right_link_dim, "Link,l=1");
		link_indices.push_back(first_right_link);
		itensor::ITensor first_site_tensor(sites(1), first_right_link);
		for(int site_val = 1; site_val <= itensor::dim(sites(1)); site_val ++){
			for(int right_link_val = 1; right_link_val < right_link_dim; right_link_val ++){
				double random_value = distribution(generator);
				new_site_tensor.set(sites(1) = site_val, first_right_link = right_link_val, random_value);
			}
		}
		psi.set(1, first_site_tensor);

		for(int i = 1; i < num_sites-1; i++){
			itensor::Index right_link(right_link_dim, "Link,l="+std::to_string(i+1));
			itensor::ITensor new_site_tensor(sites(i+1), link_indices[i-1], right_link);
			for(int site_val = 1; site_val <= itensor::dim(sites(i+1)); site_val ++){
				for(int right_link_val = 1; right_link_val < right_link_dim; right_link_val ++){
					for(int left_link_val = 1; left_link_val < left_link_dim; left_link_val ++){
						int dist = std::abs(right_link_val - left_link_val);
						double random_value = distribution(generator)*std::exp(-dist/correlation_length);
						new_site_tensor.set(sites(i+1) = site_val, link_indices[i-1] = left_link_val, right_link = right_link_val, random_value);
					}
				}
			}
			psi.set(i+1, new_site_tensor);
			link_indices.push_back(right_link);
		}
		itensor::ITensor last_site_tensor(sites(num_sites), link_indices[num_sites-2]);
		for(int site_val = 1; site_val <= itensor::dim(sites(num_sites)); site_val ++){
			for(int left_link_val = 1; left_link_val < left_link_dim; left_link_val ++){
				double random_value = distribution(generator);
				new_site_tensor.set(sites(num_sites) = site_val, link_indices[num_sites-2] = left_link_val, random_value);
			}
		}
		psi.set(num_sites, last_site_tensor);
		psi.replaceLinkInds(itensor::IndexSet(link_indices));
		return psi;
	}
}


#endif