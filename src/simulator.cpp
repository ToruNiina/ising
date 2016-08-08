#include <array>
#include <random>
#include <chrono>
#include <iostream>

#include <cmath>
#include <cstddef>
#include <cstdio>

#include <omp.h>

typedef float Real;
constexpr static std::size_t width  = 72;
constexpr static std::size_t height = 72;
constexpr static Real kB = 1.;
constexpr static Real T  = 1.;
constexpr static std::array<Real, 5> prob{{1.,
    std::exp(1.f/(-kB * T)), std::exp(2.f/(-kB * T)),
    std::exp(3.f/(-kB * T)), std::exp(4.f/(-kB * T))}};

inline bool step(const bool c,
                 const bool n, const bool e, const bool s, const bool w,
                 const Real rnd)
{
    short dE = 0;
    if(c != n) --dE; else ++dE;
    if(c != e) --dE; else ++dE;
    if(c != s) --dE; else ++dE;
    if(c != w) --dE; else ++dE;
    return (dE <= 0) ? (!c) : ((rnd < prob[dE]) != c);
}

inline void step(std::array<std::array<bool, width>, height>& space,
                 const std::size_t i, const std::size_t j, const Real rnd)
{
    space[i][j] = step(space[i][j],
       space[i][(j+1 < width) ? j+1 : j+1-width],
       space[(i+1 < height) ? i+1 : i+1-height][j],
       space[i][(j-1 >= 0) ? j-1 : j-1+width],
       space[(i-1 >= 0) ? i-1 : i-1+height][j],
       rnd);
    return;
}

inline std::array<Real, width> genrnd(std::mt19937& mt,
                                      std::uniform_real_distribution<Real>& uni)
{
    std::array<Real, width> retval;
    for(auto iter = retval.begin(); iter < retval.end(); ++iter)
        *iter = uni(mt);
    return retval;
}

inline std::array<bool, width> initline(std::mt19937& mt,
        std::bernoulli_distribution& bn)
{
    std::array<bool, width> retval;
    for(auto iter = retval.begin(); iter < retval.end(); ++iter)
        *iter = bn(mt);
    return retval;
}

int main()
{
    const std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();

    std::array<std::array<bool, width>, height> space{};
    std::mt19937 mt(10);
    std::bernoulli_distribution bn(0.5);
    std::uniform_real_distribution<Real> uni(0., 1.);

#pragma omp parallel for ordered
    for(auto iter = space.begin(); iter < space.end(); ++iter)
    {
#pragma omp ordered
        {
            *iter = initline(mt, bn);
        }//ordered
    }

    std::array<std::array<Real, width>, height> random{};
    std::size_t t = 0;

    const std::chrono::system_clock::time_point start_time =
        std::chrono::system_clock::now();
    while(t < 100)
    {
#pragma omp parallel
{
#pragma omp for ordered
        for(auto outer = random.begin(); outer < random.end(); ++outer)
        {
#pragma omp ordered
            {
            *outer = genrnd(mt, uni);
            }//ordered
        }

#pragma omp for
        for(std::size_t i=0; i<height; ++i)
        {
            if(i % 2 == 0)
                for(std::size_t j=0; j<width; j+=2)
                    step(space, i, j, random[i][j]);
            else
                for(std::size_t j=1; j<width; j+=2)
                    step(space, i, j, random[i][j]);
        }

#pragma omp for
        for(std::size_t i=0; i<height; ++i)
        {
            if(i % 2 == 0)
                for(std::size_t j=1; j<width; j+=2)
                    step(space, i, j, random[i][j]);
            else
                for(std::size_t j=0; j<width; j+=2)
                    step(space, i, j, random[i][j]);
        }

}//parallel

        for(std::size_t i=0; i<height; ++i)
        {
            for(std::size_t j=0; j<width; ++j)
                putc(space[i][j]+48, stdout);
            putc('\n', stdout);
        }
        fputc('\n', stdout);
        ++t;
    }

    const std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();

    const auto whole_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const auto time_integral =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time);

    std::cerr << "time(include initialization): " << whole_time.count() << " ms" << std::endl;
    std::cerr << "time(only time integration) : " << time_integral.count() << " ms" << std::endl;

    return EXIT_SUCCESS;
}
