#pragma once
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents 1D morse potential V(x)=D_e*(-2e^{-a(x-x0)}+e^{-2a(x-x0)})
    *
    * \tparam D dimensionality of 1D morse potential (number of variables)
    */

	template<dim_t D>
	struct Morse_zero_1D
	{
    	using RMatrixD1 = RMatrix<D, 1>;   
    	using RMatrixDD = RMatrix<D, D>; 
        using RMatrixDX = RMatrix<D, Eigen::Dynamic>;
        using RMatrix1X = RMatrix<1, Eigen::Dynamic>;

        using CMatrixXX = CMatrix<Eigen::Dynamic, Eigen::Dynamic>;
        using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
        using CMatrixX1 = CMatrix<Eigen::Dynamic, 1>;
        using CMatrixD1 = CMatrix<D, 1>;
        using CMatrixDD = CMatrix<D, D>;
        using CMatrixDX = CMatrix<D, Eigen::Dynamic>;
       


        Morse_zero_1D()= default;  // default constructor 

        Morse_zero_1D(real_t De, real_t a, real_t x0, const RMatrixDD &mass_inv): De_(De), a_(a), x0_(x0), mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        Morse_zero_1D(const Morse_zero_1D &that){

            De_=that.De_;
            a_=that.a_;
            x0_=that.x0_;
            mass_inv_=that.mass_inv_;
        }

        //assignment 
        Morse_zero_1D &operator=(const Morse_zero_1D& that){

            De_=that.De_;
            a_=that.a_;
            x0_=that.x0_;
            mass_inv_=that.mass_inv_;
            return *this;
        }

        /**
        * \brief calculate the value of 1D morse potential at position pos
        * \param[in] pos  the position at which we calculate the value of 1D morse potential
        */

    	real_t evaluate_pes(const RMatrixD1& pos) const 
    	{
            return De_*(-2.0*std::exp(-a_*(pos(0)-x0_))+std::exp(-2.0*a_*(pos(0)-x0_)));
    	}



        /**
        * \brief calculate the gradient of 1D morse potential at position pos 
        * \param[in] pos the position at which we calculate the gradient of 1D morse potential
        */

    	RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
    	{

            RMatrixD1  grad(D,1);
            
            grad(0)= 2.0*a_*De_*(std::exp(-a_*(pos(0)-x0_))-std::exp(-2.0*a_*(pos(0)-x0_)));

            return grad;
    	}

        /**
        * \brief calculate the hessian of 1D morse potential at position pos 
        * \param[in] pos the position at which we calculate the 1D morse potential
        */

    	RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
    	{
            RMatrixDD hess=RMatrixDD::Constant(0.0);

            hess(0,0)=2.0*a_*a_*De_*(-std::exp(-a_*(pos(0)-x0_))+2.0*std::exp(-2.0*a_*(pos(0)-x0_)));

            return hess;

    	}

        /**
        * \brief evaluate the values of the PES at the quadrature points nodes 
        * \param[in] nodes the quadrature points at which we calculate the values of 1D morse potential
        */

        CMatrix1X evaluate_pes_node(const CMatrixDX& nodes) const {

            dim_t n_nodes=nodes.cols();

            CMatrix1X pes(1, n_nodes);

            for(int ii=0; ii<n_nodes; ii++){
                pes(ii)=evaluate_pes_c(nodes.col(ii));
            }
            return pes;
        }


        CMatrix1X local_quadratic(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            dim_t n_nodes=nodes.cols();
            real_t pes=evaluate_pes(pos);        //calculate the potential energy at position pos 
            RMatrixD1 grad=evaluate_grad(pos);   //calculate the gradient at position pos
            RMatrixDD hess=evaluate_hess(pos);   //calculate the hessian at position pos 

            CMatrix1X Vq=RMatrix1X::Constant(1, n_nodes, pes).template cast<complex_t>();  


            CMatrixDX dx = nodes.colwise() - pos.template cast<complex_t>();   // (x-q)

            CMatrix1X grad_dx=grad.transpose()*dx;                             //\Lambda{V}(q)(x-q)

            CMatrix1X hess_dx=(0.5*dx.transpose()*hess*dx).diagonal();         ///frac(1){2}(x-q)^T\Lambda^2V(q)(x-q)

            return Vq+grad_dx+hess_dx;                                         //return the local quadratic term 
        }



    	CMatrix1X local_remainder(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            return evaluate_pes_node(nodes)-local_quadratic(nodes, pos);
        }

        // return the value of De_
        real_t & De() {

            return De_;
        }

        real_t De() const {

            return De_;
        }


        // return the value of a_
        real_t & a() {

            return a_;
        }

        real_t a() const {

            return a_;
        }

        // return the value of x0_
        real_t & x0() {

            return x0_;
        }

        real_t x0() const {

            return x0_;
        }


        // return the value of the mass_inv_
        RMatrixDD & mass_inv()
        {

            return mass_inv_;
        }

        RMatrixDD const& mass_inv() const
        {

            return mass_inv_;
        }


    private:

        real_t De_;          // the value of the De_
        real_t a_;           // the value of the a_
        real_t x0_;          // the value of the x0_
        RMatrixDD mass_inv_;      // inverse of the mass matrix M

        complex_t evaluate_pes_c(const CMatrixD1 &pos) const{

            return complex_t(De_*(-2.0*std::exp(-a_*(pos(0).real()-x0_))+std::exp(-2.0*a_*(pos(0).real()-x0_))),0.0);
        }

	};


}

