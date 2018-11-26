#include <iostream>
#include <random>
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cutf/device.hpp>
#include <cutf/error.hpp>

template <class T>
__device__ T sign(const T v){
	if( v < cutf::cuda::type::cast<T>(0.0f) ){
		return -v;
	}else{
		return v;
	}
}

template <class T>
__global__ void kernel_if(const T* const a, T* const b){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	b[tid] = sign(a[tid]);
}
template <class T>
__global__ void kernel_cutf(const T* const a, T* const b){
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	b[tid] = cutf::cuda::math::sign(a[tid]);
}

template <std::size_t N, std::size_t C, class T, class Func>
void test(Func func){
	std::cout<<__func__<<std::endl;
	auto dF = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto hF = cutf::cuda::memory::get_host_unique_ptr<T>(N);
	auto dI = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto hI = cutf::cuda::memory::get_host_unique_ptr<T>(N);
	for(auto i = decltype(N)(0); i < N; i++){
		hF.get()[i] = cutf::cuda::type::cast<T>((static_cast<float>(N/2) - i) * 10.0f);
	}

	cutf::cuda::memory::copy(dF.get(), hF.get(), N);

	for(std::size_t c = 0; c < C; c++)
		func(dF.get(), dI.get());

	cutf::cuda::memory::copy(hI.get(), dI.get(), N);
}

template <std::size_t N, class T>
void test_if(const T* const a, T* const b){
	kernel_if<T><<<N, 1>>>(a, b);
}
template <std::size_t N, class T>
void test_cutf(const T* const a, T* const b){
	kernel_cutf<T><<<N, 1>>>(a, b);
}


int main(){
	constexpr std::size_t N = 1 << 10;
	constexpr std::size_t C = 1 << 13;

	test<N, C, half>(test_if<N, half>);
	test<N, C, float>(test_if<N, float>);
	test<N, C, double>(test_if<N, double>);
	test<N, C, half>(test_cutf<N, half>);
	test<N, C, float>(test_cutf<N, float>);
	test<N, C, double>(test_cutf<N, double>);
}
