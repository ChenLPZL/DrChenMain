#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents D-dimensional Henon-Heiles potential 
    *
    * \tparam D dimensionality of Henon-Heiles potential (number of variables)
    */

	template<dim_t D>
	struct Henon_Heiles
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
       


        Henon_Heiles()= default;  // default constructor 

        Henon_Heiles(real_t lambda, const RMatrixDD &mass_inv): lambda_(lambda), mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        Henon_Heiles(const Henon_Heiles &that){

            lambda_=that.lambda_;
            mass_inv_=that.mass_inv_;
        }

        //assignment 
        Henon_Heiles &operator=(const Henon_Heiles& that){

            lambda_=that.lambda_;
            mass_inv_=that.mass_inv_;
            return *this;
        }

        /**
        * \brief calculate the value of Henon-Heiles potential at position pos
        * \param[in] pos  the position at which we calculate the value of Henon-Heiles potential
        */

    	real_t evaluate_pes(const RMatrixD1& pos) const 
    	{
            return 0.5*(pos.array()*pos.array()).sum()+lambda_*(pos.block(0,0,D-1,1).array()*pos.block(0,0,D-1,1).array()*pos.block(1,0,D-1,1).array()).sum()-lambda_/3.0*(pos.block(1,0,D-1,1).array()*pos.block(1,0,D-1,1).array()*pos.block(1,0,D-1,1).array()).sum();
    	}



        /**
        * \brief calculate the gradient of the Henon-Heiles potential  at position pos 
        * \param[in] pos the position at which we calculate the gradient of Henon-Heiles potential
        */

    	RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
    	{
            /*******************************************************************
            for(int ii=1; ii<ndims-1; ii++)
                grad[ii]=qq[ii]+lambda*(qq[ii-1]*qq[ii-1]-qq[ii]*qq[ii]+2.0*qq[ii]*qq[ii+1]);
        
            grad[0]=qq[0]+2.0*lambda*qq[0]*qq[1];
            grad[ndims-1]=qq[ndims-1]+lambda*(qq[ndims-2]*qq[ndims-2]-qq[ndims-1]*qq[ndims-1]);
            ********************************************************************/

            RMatrixD1  grad(D,1);
            for(int ii=1; ii<int(D-1); ii++){
                grad(ii)=pos(ii)+lambda_*(pos(ii-1)*pos(ii-1)-pos(ii)*pos(ii)+2.0*pos(ii)*pos(ii+1));
            }

            grad(0)=pos(0)+2.0*lambda_*pos(0)*pos(1);
            grad(D-1)=pos(D-1)+lambda_*(pos(D-2)*pos(D-2)-pos(D-1)*pos(D-1));

            return grad;
    	}

        /**
        * \brief calculate the hessian of the Henon-Heiles potential  at position pos 
        * \param[in] pos the position at which we calculate the hessian of Henon-Heiles potential
        */

    	RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
    	{
            RMatrixDD hess=RMatrixDD::Constant(0.0);

            for(int ii=1; ii<int(D-1); ii++){
                hess(ii,ii-1)=2.0*lambda_*pos(ii-1);
                hess(ii,ii)=1.0+lambda_*(-2.0*pos(ii)+2.0*pos(ii+1));
                hess(ii,ii+1)=2.0*lambda_*pos(ii);
            }
            hess(0, 0)=1.0+2.0*lambda_*pos(1);
            hess(0, 1)=2.0*lambda_*pos(0);
            hess(D-1,D-2)=2.0*lambda_*pos(D-2);
            hess(D-1,D-1)=1.0-2.0*lambda_*pos(D-1);

            return hess;

    	}

        /**
        * \brief evaluate the values of the PES at the quadrature points nodes 
        * \param[in] nodes the quadrature points at which we calculate the values of Henon-Heiles potential
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

        // return the value of lambda_
        real_t & lambda() {

            return lambda_;
        }

        real_t lambda() const {

            return lambda_;
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

        real_t lambda_;          // the mode-mode coupling strength 
        RMatrixDD mass_inv_;      // inverse of the mass matrix M

        complex_t evaluate_pes_c(const CMatrixD1 &pos) const{

            return 0.5*(pos.array()*pos.array()).sum()+lambda_*(pos.block(0,0,D-1,1).array()*pos.block(0,0,D-1,1).array()*pos.block(1,0,D-1,1).array()).sum()-lambda_/3.0*(pos.block(1,0,D-1,1).array()*pos.block(1,0,D-1,1).array()*pos.block(1,0,D-1,1).array()).sum();
        }

	};


}

