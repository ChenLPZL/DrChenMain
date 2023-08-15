#pragma once
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents 2D quartic oscillator V(x)=\sigma_x*x^4+\sigma_y*y^4
    *
    * \tparam D dimensionality of 2D quartic oscillator (number of variables)
    */

	template<dim_t D>
	struct Quartic_2D
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
       


        Quartic_2D()= default;  // default constructor 

        Quartic_2D(real_t sigmax, real_t sigmay, const RMatrixDD &mass_inv): sigmax_(sigmax), sigmay_(sigmay), mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        Quartic_2D(const Quartic_2D &that){

            sigmax_=that.sigmax_;
            sigmay_=that.sigmay_;
            mass_inv_=that.mass_inv_;
        }

        //assignment 
        Quartic_2D &operator=(const Quartic_2D& that){

            sigmax_=that.sigmax_;
            sigmay_=that.sigmay_;
            mass_inv_=that.mass_inv_;
            return *this;
        }

        /**
        * \brief calculate the value of 2D quartic oscillator at position pos
        * \param[in] pos  the position at which we calculate the value of 2D quartic oscillator
        */

    	real_t evaluate_pes(const RMatrixD1& pos) const 
    	{
            return sigmax_*pos(0)*pos(0)*pos(0)*pos(0)+sigmay_*pos(1)*pos(1)*pos(1)*pos(1);
    	}



        /**
        * \brief calculate the gradient of 2D quartic oscillator at position pos 
        * \param[in] pos the position at which we calculate the gradient of 2D quartic oscillator
        */

    	RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
    	{

            RMatrixD1  grad(D,1);
            
            grad(0)=4.0*sigmax_*pos(0)*pos(0)*pos(0);
            grad(1)=4.0*sigmay_*pos(1)*pos(1)*pos(1);

            return grad;
    	}

        /**
        * \brief calculate the hessian of 2D quartic oscillator at position pos 
        * \param[in] pos the position at which we calculate the 2D quartic oscillator
        */

    	RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
    	{
            RMatrixDD hess=RMatrixDD::Constant(0.0);

            hess(0,0)=12.0*sigmax_*pos(0)*pos(0);
            hess(0,1)=0.0;
            hess(1,0)=0.0;
            hess(1,1)=12.0*sigmay_*pos(1)*pos(1);

            return hess;

    	}

        /**
        * \brief evaluate the values of the PES at the quadrature points nodes 
        * \param[in] nodes the quadrature points at which we calculate the values of 2D harmonic oscillator
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

        // return the value of sigma_x
        real_t & sigmax() {

            return sigmax_;
        }

        real_t sigmax() const {

            return sigmax_;
        }

        // return the value of sigma_y
        real_t & sigmay() {

            return sigmay_;
        }

        real_t sigmay() const {

            return sigmay_;
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

        real_t sigmax_;          // the value of sigmax_
        real_t sigmay_;          // the value of sigmay_
        RMatrixDD mass_inv_;      // inverse of the mass matrix M

        complex_t evaluate_pes_c(const CMatrixD1 &pos) const{

            return sigmax_*pos(0)*pos(0)*pos(0)*pos(0)+sigmay_*pos(1)*pos(1)*pos(1)*pos(1);
        }

	};


}

