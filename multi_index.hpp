#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include "types.hpp"


template<dim_t D>
std::ostream &operator<<(std::ostream &out, const TinyMultiIndex<D> &index)
{
    std::cout << "(";
    int size=index.size();
    for(int ii=0; ii<size-1; ii++)
        std::cout<<index[ii]<<",";
    std::cout<<index[size-1];
    std::cout<<")";

    return out;
}

namespace std {

  /**
     * \cond HIDDEN_SYMBOLS
     * Provides less functor (compare) for STL containers (notable std::map).
     * Specializes generic std::less<T>.
     * \endcond
     */

    template<dim_t D>
    class less< TinyMultiIndex<D> >
    {
    private:
        typedef TinyMultiIndex<D> MultiIndex;

    public:
        typedef MultiIndex first_argument_type;
        typedef MultiIndex second_argument_type;
        typedef bool result_type;

        
        bool operator()( MultiIndex const& first,  MultiIndex const& second) const 
        {
            return lexicographical_compare(first.begin(), first.end(), second.begin(), second.end());

        }

    };


    /**
     * \cond HIDDEN_SYMBOLS
     * Provides hash functor for STL containers (notable std::unordered_map).
     * Specializes generic std::hash<T>.
     * \endcond
     */

    template<dim_t D>
    class hash< TinyMultiIndex<D> >
    {
    private:
        typedef TinyMultiIndex<D> MultiIndex;

        std::string  to_str( MultiIndex const& index) const 
        {
             std::stringstream ss;
             for(auto ii: index)
                ss<<ii;

             return ss.str();
        }

    public:

        std::size_t operator()(MultiIndex const& index) const
        {

            return std::hash<std::string>()(to_str(index));

        }
        
    };

    /**
     * \cond HIDDEN_SYMBOLS
     * Provides equality functor for STL containers (notable std::unordered_map).
     * Specializes generic std::equal_to<T>.
     * \endcond
     */

    template<dim_t D>
    class equal_to< TinyMultiIndex<D> >
    {
    private:
        typedef TinyMultiIndex<D> MultiIndex;

        std::string  to_str( MultiIndex const& index) const 
        {
             std::stringstream ss;
             for(auto ii: index)
                ss<<ii;

             return ss.str();
        }

    public:
        typedef MultiIndex first_argument_type;
        typedef MultiIndex second_argument_type;
        typedef bool result_type;   


        
        bool operator()(MultiIndex const& first, MultiIndex const& second) const 
        {
            std::string first_str=to_str(first);
            std::string second_str=to_str(second);

            return first_str==second_str;       
        }     
        
    };

}


