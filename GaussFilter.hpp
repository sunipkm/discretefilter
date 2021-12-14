/**
 * @file GaussFilter.hpp
 * @author Sunip K. Mukherjee (sunipkmukherjee@gmail.com)
 * @brief Gaussian filter library, applies Gauss filter on data in a Ring Buffer
 * @version 0.1
 * @date 2021-12-14
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef _GAUSS_FILTER_HPP_
#define _GAUSS_FILTER_HPP_

#include <stdexcept>
#include <cmath>

#include "RingBuf.hpp"

template <class T>
class GaussFilter
{
private:
    T *gauss_coeff;
    int size;

    T factorial(int i)
    {
        T result = 1;
        if (i == 0)
            return 1;
        do
        {
            result *= i;
        } while (--i > 0);
        return result;
    }

public:
    GaussFilter() : gauss_coeff(nullptr)
    {
    }
    ~GaussFilter()
    {
        if (gauss_coeff != nullptr)
            delete[] gauss_coeff;
    }
    GaussFilter(int size, float sigma)
    {
        if (size < 2)
            throw std::invalid_argument("Bessel filter size can not be < 2.");
        gauss_coeff = new T[size];
        this->size = size;
        for (int i = 0; i < size; i++)
        {
            gauss_coeff[i] = exp(-(i * i / sigma / sigma));
        }
    }
    template <class U>
    U ApplyFilter(RingBuf<U> &buf)
    {
        if (!buf.HasData())
        {
            throw std::invalid_argument("Ring buffer not initialized");
        }
        U val = 0;
        U coeff_sum = 0;
        int max_lim = buf.GetSize() > size ? size : buf.GetSize();
        for (int i = 0;;)
        {
            val += gauss_coeff[i] * buf[i];
            coeff_sum += gauss_coeff[i];
            i++;
            if (gauss_coeff[i] < 0.001 || i >= size || i >= (buf.GetPushed() % buf.GetSize()))
                break;
        }
        return val / coeff_sum;
    }
    template <class U>
    U ApplyFilter(RingBuf<U> *&buf)
    {
        return ApplyFilter(*buf);
    }
};

#endif // _BESSEL_FILTER_HPP_