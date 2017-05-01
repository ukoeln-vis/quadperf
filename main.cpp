// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iostream>
#include <random>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/aligned_vector.h>
#include <visionaray/array.h>
#include <common/timer.h>
#include <visionaray/traverse.h>

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
#define ALIGNMENT 64
#elif VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
#define ALIGNMENT 32
#elif VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE)
#define ALIGNMENT 16
#endif

#ifndef ALIGNMENT
#define ALIGNMENT 32
#endif

#define CALCULATE_UV 0

#define QUAD_NS visionaray

#include "swoop.h"
#include "opt.h"
#include "basic_quad.h"

using namespace visionaray;


#ifdef __CUDACC__
template
<typename Intersector, typename quad_type, typename ray_type>
__global__ void cuda_kernel(ray_type *rays, unsigned int ray_count, quad_type *first, quad_type *last, float *output_ts)
{
    Intersector i;
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < ray_count)
    {
        ray_type r = rays[index];
        auto hr = closest_hit(r, first, last, i);
        output_ts[index] = hr.t;
    }
}
#endif

struct benchmark
{
    typedef basic_quad<float> quad_type;
    typedef quad_prim<float> quad_type_opt;
    typedef swoop_quad<float> quad_type_swoop;
    typedef basic_ray<float> ray_type;

    aligned_vector<quad_type_opt, ALIGNMENT> quads_opt;
    aligned_vector<quad_type_swoop, ALIGNMENT> quads_swoop;
    aligned_vector<quad_type, ALIGNMENT> quads;
    aligned_vector<ray_type, ALIGNMENT> rays;

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE)
    aligned_vector<basic_ray<simd::float4>, ALIGNMENT> rays_cpu4;
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
    aligned_vector<basic_ray<simd::float8>, ALIGNMENT> rays_cpu8;
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
    aligned_vector<basic_ray<simd::float16>, ALIGNMENT> rays_cpu16;
#endif

    std::string name_;
    bool cuda_test;
    int cuda_block_size = 192;
    int cpu_packet_size = 1;


    // const unsigned int quad_count = 100000;
    // const unsigned int ray_count = (1<<18);
    // const unsigned int quad_count = 10000;
    // const unsigned int ray_count = (1<<16);
    const unsigned int quad_count = 1000;
    const unsigned int ray_count = (1<<14);


    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<float> uniform_dist;

    rand_engine  rng;
    uniform_dist dist;

    benchmark(std::string name, bool cuda_test)
        : name_(name)
        , cuda_test(cuda_test)
        , rng(0)
        , dist(0, 1)
    {
        generate_quads();
        generate_rays();
    }

    void generate_quads()
    {
        for (size_t i=0; i<quad_count; i++)
        {
            vec3 u;
            vec3 v;
            vec3 w = normalize(vec3(dist(rng), dist(rng), dist(rng)));
            make_orthonormal_basis(u, v, w);

            //vec3 center = vec3(vec3(dist(rng), dist(rng), dist(rng)));
            vec3 center = vec3(0.f);

            quad_type quad = quad_type::make_quad(
                    u * dist(rng) + center,
                    v * dist(rng) + center,
                    -u * dist(rng) + center,
                    -v * dist(rng) + center
                    );
            quads.push_back(quad);


            quads_opt.push_back(quad_type_opt::make_quad(
                    u * dist(rng) + center,
                    v * dist(rng) + center,
                    -u * dist(rng) + center,
                    -v * dist(rng) + center
                    ));

            quads_swoop.push_back(quad_type_swoop::make_quad(
                    u * dist(rng) + center,
                    v * dist(rng) + center,
                    -u * dist(rng) + center,
                    -v * dist(rng) + center
                    ));
        }
    }

    void generate_rays()
    {
        for (size_t i=0; i<ray_count; i++)
        {
            ray r;

            vec3 origin(0.f, 0.f, 4.f);
            vec3 dir = normalize(vec3(dist(rng), dist(rng), 0.f) - origin);

            r.ori = origin;
            r.dir = dir;

            rays.push_back(r);
        }
    }

    template <typename V1, typename V2>
    void pack_rays(V1& rays_cpu, V2 const& rays)
    {
        const size_t packet_size = simd::num_elements<typename V1::value_type::scalar_type>::value;

        assert(rays.size() % packet_size == 0);

        for (size_t i=0; i<rays.size()/packet_size; i++)
        {
            array<typename V2::value_type, packet_size> ra;

            for (size_t e=0; e<packet_size; ++e)
            {
                ra[e] = rays[i*packet_size + e];
            }

            rays_cpu.push_back(simd::pack(ra));
        }
    }

    void init()
    {
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE)
        pack_rays(rays_cpu4, rays);
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
        pack_rays(rays_cpu8, rays);
#endif
#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
        pack_rays(rays_cpu16, rays);
#endif

#ifdef __CUDACC__
        init_device_data();
#endif
    }


    double operator()()
    {
        if (!cuda_test)
        {
            if (name_ == "opt")
                return run_test<quad_intersector_opt>(quads_opt);
            if (name_ == "mt bl uv")
                return run_test<quad_intersector_mt_bl_uv>(quads);
            if (name_ == "pluecker")
                return run_test<quad_intersector_pluecker>(quads);
            if (name_ == "project 2d")
                return run_test<quad_intersector_project_2D>(quads);
            if (name_ == "uv")
                return run_test<quad_intersector_uv>(quads);
            if (name_ == "swoop")
                return run_test<quad_intersector_swoop>(quads_swoop);
        }
        else
        {
#ifdef __CUDACC__
            return run_cuda_test();
#endif
        }
        return 0.0;
    }

    template <typename intersector, typename QT>
    double run_test(aligned_vector<QT, ALIGNMENT> &quads)
    {
        intersector i;

        if (cpu_packet_size == 1)
        {
            timer t;

            //#pragma omp parallel for
            for (size_t j = 0; j < rays.size(); ++j)
            {
                volatile auto hr = closest_hit(rays[j], quads.begin(), quads.end(), i);
            }

            return t.elapsed();
        }

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_SSE)
        else if (cpu_packet_size == 4)
        {
           timer t;

           //#pragma omp parallel for
           for (size_t j = 0; j < rays_cpu4.size(); ++j)
           {
               volatile auto hr = closest_hit(rays_cpu4[j], quads.begin(), quads.end(), i);
           }

           return t.elapsed();
        }
#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX)
        else if (cpu_packet_size == 8)
        {
            timer t;

            //#pragma omp parallel for
            for (size_t j = 0; j < rays_cpu8.size(); ++j)
            {
                volatile auto hr = closest_hit(rays_cpu8[j], quads.begin(), quads.end(), i);
            }

            return t.elapsed();
        }
#endif

#if VSNRAY_SIMD_ISA_GE(VSNRAY_SIMD_ISA_AVX512F)
        else if (cpu_packet_size == 16)
        {
            timer t;

            //#pragma omp parallel for
            for (size_t j = 0; j < rays_cpu16.size(); ++j)
            {
                volatile auto hr = closest_hit(rays_cpu16[j], quads.begin(), quads.end(), i);
            }

            return t.elapsed();
        }
#endif

        return 0;
    }

#ifdef __CUDACC__
    thrust::device_vector<quad_type> d_quads;
    thrust::device_vector<quad_type_opt> d_quads_opt;
    thrust::device_vector<quad_type_swoop> d_quads_swoop;
    thrust::device_vector<ray_type> d_rays;

    thrust::device_vector<float> output_ts;

    void init_device_data()
    {
        d_quads = thrust::device_vector<quad_type>(quads);
        d_rays = thrust::device_vector<ray_type>(rays);
        d_quads_opt = thrust::device_vector<quad_type_opt>(quads_opt);
        d_quads_swoop = thrust::device_vector<quad_type_swoop>(quads_swoop);

        output_ts = thrust::device_vector<float>(rays.size());
    }

    double run_cuda_test()
    {
        dim3 block_size(cuda_block_size);
        dim3 grid_size(div_up(ray_count, block_size.x));

        // int min_block_size;
        // int min_grid_size;
        //
        // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &min_block_size, cuda_kernel<quad_intersector_pluecker, quad_type, ray_type>, 0, 0);
        // std::cout << min_block_size << std::endl << min_grid_size << std::endl;

        if (name_ == "opt")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_opt> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads_opt.data()),
                    thrust::raw_pointer_cast(d_quads_opt.data()) + d_quads_opt.size(),
                    thrust::raw_pointer_cast(output_ts.data())
                    );

            return t.elapsed();
        }
        else if (name_ == "mt bl uv")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_mt_bl_uv> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size(),
                    thrust::raw_pointer_cast(output_ts.data())
                    );

            return t.elapsed();
        }
        else if (name_ == "pluecker")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_pluecker> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size(),
                    thrust::raw_pointer_cast(output_ts.data())
                    );

            return t.elapsed();
        }
        else if (name_ == "project 2d")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_project_2D> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size(),
                    thrust::raw_pointer_cast(output_ts.data())
                    );

            return t.elapsed();
        }
        else if (name_ == "uv")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_uv> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size(),
                    thrust::raw_pointer_cast(output_ts.data())
                    );

            return t.elapsed();
        }
        else if (name_ == "swoop")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_swoop> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads_swoop.data()),
                    thrust::raw_pointer_cast(d_quads_swoop.data()) + d_quads_swoop.size(),
                    thrust::raw_pointer_cast(output_ts.data())
                    );

            return t.elapsed();
        }

        return 0.0;
    }
#endif
};


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    int dry_runs = 3;   // some dry runs to fill the caches
    int bench_runs = 10;
    int bs = 192;
    int cpu_packet_size = 1;

    int do_cuda_test = 0;

    std::string name = "opt";


    using namespace support;

    cl::CmdLine cmd;

    auto bsref = cl::makeOption<int&>(
            cl::Parser<>(), cmd, "blocksize",
            cl::ArgName("blocksize"),
            cl::ArgRequired,
            cl::init(bs),
            cl::Desc("CUDA block size")
            );

    auto nref = cl::makeOption<std::string&>(
            cl::Parser<>(), cmd, "intersect",
            cl::ArgName("intersect"),
            cl::ArgRequired,
            cl::init(name),
            cl::Desc("Intersection algorithm")
            );

    auto ctref = cl::makeOption<int&>(
            cl::Parser<>(), cmd, "cuda_test",
            cl::ArgName("cuda_test"),
            cl::init(do_cuda_test),
            cl::Desc("Whether to test cuda algorithm")
            );

    auto psref = cl::makeOption<int&>(
            cl::Parser<>(), cmd, "cpu_packet_size",
            cl::ArgName("cpu_packet_size"),
            cl::init(cpu_packet_size),
            cl::Desc("CPU simd packet size (1,4,8,16)")
            );

    try
    {
        auto args = std::vector<std::string>(argv + 1, argv + argc);

        cl::expandWildcards(args);
        //cl::expandResponseFiles(args, cl::TokenizeWindows());
        cl::expandResponseFiles(args, cl::TokenizeUnix());

        cmd.parse(args);
    }
    catch (std::exception& e)
    {
        std::cout << "error: " << e.what() << '\n';
        std::cout << '\n';
        std::cout << cmd.help("benchmark") << '\n';
        return -1;
    }

    benchmark b(name, do_cuda_test);
    b.init();
    b.cuda_block_size = bs;
    b.cpu_packet_size = cpu_packet_size;

    for (int i = 0; i < dry_runs; ++i)
    {
        volatile double t = b() * 1000.0;
    }

    std::vector<double> times(bench_runs);
    for (int i = 0; i < bench_runs; ++i)
    {
        times[i] = b() * 1000.0;
    }

    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    std::cout << "Benchmark:   " << name << '\n';
    if (do_cuda_test)
    std::cout << "CUDA grid:   " << div_up((int)b.rays.size(), b.cuda_block_size) << " blocks of size " << b.cuda_block_size << '\n';
    else
    std::cout << "SIMD Packet: " << cpu_packet_size << '\n';
    std::cout << "Num quads:   " << b.quads.size() << '\n';
    std::cout << "Num rays:    " << b.rays.size() << '\n';
    std::cout << "Rays/sec:    " << b.rays.size() * bench_runs * 1000.0 / sum << '\n';
    std::cout << "Average:     " << sum / bench_runs << " ms\n";
    std::cout << "Median:      " << times[bench_runs / 2] << " ms\n";
    std::cout << "Max:         " << times.back() << " ms\n";
    std::cout << "Min:         " << times[0] << " ms\n";
    std::cout << std::endl;

    return 0;
}
