#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents 1D double well  V(x)=a*x^4-b*x^2
    *
    * \tparam D dimensionality of 1D double well (number of variables)
    */

	template<dim_t D>
	struct Double_well2_1D
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
       


        Double_well2_1D()= default;  // default constructor 

        Double_well2_1D(real_t a, real_t b, const RMatrixDD &mass_inv): a_(a), b_(b), mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        Double_well2_1D(const Double_well2_1D &that){

            a_=that.a_;
            b_=that.b_;
            mass_inv_=that.mass_inv_;
        }

        //assignment 
        Double_well2_1D &operator=(const Double_well2_1D& that){

            a_=that.a_;
            b_=that.b_;
            mass_inv_=that.mass_inv_;
            return *this;
        }

        /**
        * \brief calculate the value of 1D double well at position pos
        * \param[in] pos  the position at which we calculate the value of 1D double well
        */

    	real_t evaluate_pes(const RMatrixD1& pos) const 
    	{
            return a_*pos(0)*pos(0)*pos(0)*pos(0)-b_*pos(0)*pos(0);
    	}



        /**
        * \brief calculate the gradient of 1D double well at position pos 
        * \param[in] pos the position at which we calculate the gradient of 1D double well 
        */

    	RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
    	{

            RMatrixD1  grad(D,1);
            
            grad(0)=4.0*a_*pos(0)*pos(0)*pos(0)-2.0*b_*pos(0);

            return grad;
    	}

        /**
        * \brief calculate the hessian of 1D double well at position pos 
        * \param[in] pos the position at which we calculate the 1D double well 
        */

    	RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
    	{
            RMatrixDD hess=RMatrixDD::Constant(0.0);

            hess(0,0)=12.0*a_*pos(0)*pos(0)-2.0*b_;

            return hess;

    	}

        /**
        * \brief evaluate the values of the PES at the quadrature points nodes 
        * \param[in] nodes the quadrature points at which we calculate the values of 1D double well 
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

        // return the value of a_
        real_t & a() {

            return a_;
        }

        real_t a() const {

            return a_;
        }


        // return the value of b_
        real_t & b() {

            return b_;
        }

        real_t b() const {

            return b_;
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

        real_t a_;          // the value of the a_
        real_t b_;          // the value of the b_
        RMatrixDD mass_inv_;      // inverse of the mass matrix M

        complex_t evaluate_pes_c(const CMatrixD1 &pos) const{

            return a_*pos(0)*pos(0)*pos(0)*pos(0)-b_*pos(0)*pos(0); 
        }

	};


}

