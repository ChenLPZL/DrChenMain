#pragma once
#include "potentialsLib.hpp"
#include "../types.hpp"


namespace potentials
{	
    /**
    * \brief This class represents a scalar potential V(x)
    *
    * \tparam D dimensionality of PES (number of variables)
    * \tparam PES specific potential experession providing the potential value, gradient and hessian at certain point 
    */

    template<dim_t D, class PES>
    class MatrixPotential1S{
    private:
    	PES pes_;       // scalar potential V(x) (see potentialLib.hpp for various scalar potential V(x))

    public:

    	MatrixPotential1S() = default;  // default constructor 

    	//Constructor 
    	MatrixPotential1S(const PES& pes)
    	{
    		pes_=pes;
    	}

        //copy constructor 
        MatrixPotential1S(const MatrixPotential1S& that){ 
            
        	pes_=that.pes_;
        }
    
        MatrixPotential1S &operator=(const MatrixPotential1S& that){  //assignment operator
            
 			pes_=that.pes_;
            return *this;
        }

        /**
        * \brief Grants access to the scalar potential surface 
        */
        PES & pes()
        {
        	return pes_;
        }

        PES const& pes() const
        {
            return pes_;
        } 
         
    };


    /**
    * \brief This class represents a matrix potential 
    *
    * \tparam D dimensionality of PES (number of variables)
    * \tparam PES specific matrix potential experession providing the potential value, gradient and hessian at certain point and local remainder 
    */

    template<dim_t D, class PES>
    class MatrixPotentialMS{
    private:
        PES pes_;       // scalar potential V(x) (see potentialLib.hpp for various scalar potential V(x))

    public:

        MatrixPotentialMS() = default;  // default constructor 

        //Constructor 
        MatrixPotentialMS(const PES& pes)
        {
            pes_=pes;
        }

        //copy constructor 
        MatrixPotentialMS(const MatrixPotentialMS& that){ 
            
            pes_=that.pes_;
        }
    
        MatrixPotentialMS &operator=(const MatrixPotentialMS& that){  //assignment operator
            
            pes_=that.pes_;
            return *this;
        }

        /**
        * \brief Grants access to the matrix potential 
        */
        PES & pes()
        {
            return pes_;
        }

        PES const& pes() const
        {
            return pes_;
        } 
         
    };

}


