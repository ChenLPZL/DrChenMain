#pragma once
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents D-dimensional torsional potential V(x)=\sum_{j=1}^D(1-cos(x_j))
    *
    * \tparam D dimensionality of D-dimensional torsional potential(number of variables)
    */

	template<dim_t D>
	struct Torsion_XD
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
       

        Torsion_XD()= default;  // default constructor 

        Torsion_XD(const RMatrixDD &mass_inv): mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        Torsion_XD(const Torsion_XD &that){

            mass_inv_=that.mass_inv_;
        }

        //assignment 
        Torsion_XD &operator=(const Torsion_XD& that){

            mass_inv_=that.mass_inv_;
            return *this;
        }



        /**
        * \brief calculate the value of D-dimensional torsional potential at position pos
        * \param[in] pos  the position at which we calculate the value of D-dimensional torsional potential
        */

    	real_t evaluate_pes(const RMatrixD1& pos) const 
    	{

            return D*1.0-Eigen::cos(pos.array()).sum();
    	}



        /**
        * \brief calculate the gradient of D-dimensional torsional potential at position pos 
        * \param[in] pos the position at which we calculate the gradient of D-dimensional torsional potential
        */

    	RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
    	{

            RMatrixD1  grad(D,1);

            for(int ii=0; ii<int(D); ii++){
                grad(ii)=std::sin(pos(ii));
            }
            
            return grad;
    	}

        /**
        * \brief calculate the hessian of D-dimensional torsional potential at position pos 
        * \param[in] pos the position at which we calculate the D-dimensional torsional potential
        */

    	RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
    	{
            RMatrixDD hess=RMatrixDD::Constant(0.0);

            for(int ii=0; ii<int(D); ii++){
                hess(ii, ii)=std::cos(pos(ii));
            }

            return hess;

    	}

        /**
        * \brief evaluate the values of the PES at the quadrature points nodes 
        * \param[in] nodes the quadrature points at which we calculate the values of 1D cosin oscillator
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

        RMatrixDD mass_inv_;      // inverse of the mass matrix M

        complex_t evaluate_pes_c(const CMatrixD1 &pos) const{

            return complex_t(D*1.0-Eigen::cos(pos.real().array()).sum(),0.0);
        }

	};


}

