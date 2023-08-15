#pragma once
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents a 4D pyrazine model 
    *
    * \tparam D dimensionality of 4D pyrazine model (number of variables)
    *
    * \tparam N energy level of the 4D pyrazine model (number of the PES)
    */

  
    const double lambda=0.144641774101584;    // hatree^{1/2}
    const double Delta=0.016967162048500;     // hatree
    

    //frequency of omega_l (no unit) [10a, 6a, 1, 9a]
    const RVector<4> omega(
    (RVector<4>() <<
        1.264864864864865*1.264864864864865,
        1.0,
        1.720270270270270*1.720270270270270,
        2.118918918918919*2.118918918918919
    ).finished());

    //frequency of kappa1_l (hatree^{1/2}) [10a, 6a, 1, 9a]
    const RVector<4> kappa1(
    (RVector<4>() <<
        0.0,
        -0.067933789407902,
        0.043441519397318,
        0.163513755411313
    ).finished());

    //frequency of kappa2_l (hatree^{1/2}) [10a, 6a, 1, 9a]
    const RVector<4> kappa2(
    (RVector<4>() <<
         0.0,
         0.084142058664975,
         0.185966674526393,
         0.049649095118617
    ).finished());



    template<dim_t D, dim_t N>
    struct pyrazine_4D
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

        pyrazine_4D()= default;  // default constructor 

        pyrazine_4D(dim_t leading_order, const RMatrixDD &mass_inv): leading_order_(leading_order), mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        pyrazine_4D(const pyrazine_4D &that){

            leading_order_=that.leading_order_;
            mass_inv_=that.mass_inv_;
        }

        //assignment 
        pyrazine_4D &operator=(const pyrazine_4D& that){

            leading_order_=that.leading_order_;
            mass_inv_=that.mass_inv_;
            return *this;
        }

        //evaluate the value of V_{ij} at the qudrature node: nodes 
        CMatrix1X evaluate_pes_node(const CMatrixDX& nodes, dim_t ii, dim_t jj) const {

            dim_t n_nodes=nodes.cols();

            CMatrix1X pes(1, n_nodes);

            //the case of V_{00}
            if(ii==0&& jj==0){
                for(int nn=0; nn<n_nodes; nn++){
                    pes(nn)=evaluate_pes_c00(nodes.col(nn));
                }
            }

            //the case of V_{01}
            if(ii==0&& jj==1){
                for(int nn=0; nn<n_nodes; nn++){
                    pes(nn)=evaluate_pes_c01(nodes.col(nn));
                }
            }

            //the case of V_{10}
            if(ii==1&& jj==0){
                for(int nn=0; nn<n_nodes; nn++){
                    pes(nn)=evaluate_pes_c10(nodes.col(nn));
                }
            }

            //the case of V_{11}
            if(ii==1&& jj==1){
                for(int nn=0; nn<n_nodes; nn++){
                    pes(nn)=evaluate_pes_c11(nodes.col(nn));
                }
            }

            return pes;
        }

        //evaluate the value of the energy level chi_i
        real_t evaluate_pes(const RMatrixD1& pos, dim_t ii) const 
        {
            
            if(ii==0){
                return evaluate_pes_chi0(pos);
            }else{
                return evaluate_pes_chi1(pos);
            }
        }

        //evaluate the value of the energy level chi_{leading_order}
        real_t evaluate_pes(const RMatrixD1& pos) const 
        {
            
            if(leading_order_==0){
                return evaluate_pes_chi0(pos);
            }else{
                return evaluate_pes_chi1(pos);
            }
        }


        //evaluate the gradient of energy level chi_i
        RMatrixD1 evaluate_grad(const RMatrixD1& pos, dim_t ii) const 
        {

            if(ii==0){
                return evaluate_grad_chi0(pos);
            }else{
                return evaluate_grad_chi1(pos);
            }
        }

        //evaluate the gradient of energy level chi_{leading_order}
        RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
        {

            if(leading_order_==0){
                return evaluate_grad_chi0(pos);
            }else{
                return evaluate_grad_chi1(pos);
            }
        }


        //evaluate the hessian of energy level chi_i
        RMatrixDD evaluate_hess(const RMatrixD1& pos, dim_t ii) const 
        {
            
            if(ii==0){
                return evaluate_hess_chi0(pos);
            }else{
                return evaluate_hess_chi1(pos);
            }

        }

        //evaluate the hessian of energy level chi_{leading_order}
        RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
        {
            
            if(leading_order_==0){
                return evaluate_hess_chi0(pos);
            }else{
                return evaluate_hess_chi1(pos);
            }

        }

        //evaluate the local remainder of the vector PES
        CMatrix1X local_remainder(const CMatrixDX& nodes, const RMatrixD1& pos, dim_t ii, dim_t jj) const {
            
            //the case of V_{00}
            if(ii==0&& jj==0){
                
                return local_remainder_00(nodes, pos);
            }

            //the case of V_{01}
            if(ii==0&& jj==1){
                
                return evaluate_pes_node(nodes, ii, jj);
            }

            //the case of V_{10}
            if(ii==1&& jj==0){
                
                return evaluate_pes_node(nodes, ii, jj);
            }

            //the case of V_{11}
            if(ii==1&& jj==1){
                
                return local_remainder_11(nodes, pos);
            }
            
        }

        //evaluate the local remainder of the vector PES
        CMatrix1X local_remainder_homogenous(const CMatrixDX& nodes, const RMatrixD1& pos, dim_t ii, dim_t jj) const {
            
            //the case of V_{00}
            if(ii==0&& jj==0){
                
                return local_remainder_00_homogenous(nodes, pos);
            }

            //the case of V_{01}
            if(ii==0&& jj==1){
                
                return evaluate_pes_node(nodes, ii, jj);
            }

            //the case of V_{10}
            if(ii==1&& jj==0){
                
                return evaluate_pes_node(nodes, ii, jj);
            }

            //the case of V_{11}
            if(ii==1&& jj==1){
                
                return local_remainder_11_homogenous(nodes, pos);
            }
            
        }

        // return the value of leading_order_
        dim_t & leading_order()
        {

            return leading_order_;
        }

        dim_t const& leading_order() const
        {

            return leading_order_;
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
        dim_t     leading_order_;       // the leading order of the N-level PES (used for the propagation of the homogenous Hagedorn wavepackets)
        RMatrixDD mass_inv_;          // inverse of the mass matrix M

        //check whether the index ii equalTo to jj 
        real_t equalTo(int ii, int jj) const {

        	if(ii==jj){
        		return 1.0;
        	}else{
        		return 0.0;
        	}
        }

        //evaluate the value of V_{00} at position pos
        complex_t evaluate_pes_c00(const CMatrixD1 &pos) const{

            complex_t result=complex_t(-Delta, 0.0);
            for(dim_t ii=0; ii<D; ii++){
                result=result+omega(ii)*pos(ii)*pos(ii)/2.0+kappa1(ii)*pos(ii);
            }
            return result;
        }

        //evaluate the value of V_{01} at position pos
        complex_t evaluate_pes_c01(const CMatrixD1 &pos) const{

            return complex_t(lambda*pos(0).real(), 0.0);
        }

        //evaluate the value of V_{10} at position pos 
        complex_t evaluate_pes_c10(const CMatrixD1 &pos) const{

            return complex_t(lambda*pos(0).real(), 0.0);
        }

        //evaluate the value of V_{11} at position pos
        complex_t evaluate_pes_c11(const CMatrixD1 &pos) const{
            
            complex_t result=complex_t(Delta, 0.0);
            for(dim_t ii=0; ii<D; ii++){
                result=result+omega(ii)*pos(ii)*pos(ii)/2.0+kappa2(ii)*pos(ii);
            }
            return result;
        }


        //evaluate the value of energy level chi_0 at position pos 
        real_t evaluate_pes_chi0(const RMatrixD1& pos) const 
        {
            return 0.5*(omega.array()*pos.array()*pos.array()+(kappa1.array()+kappa2.array())*pos.array()).sum()-0.5*std::sqrt(std::pow(2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum(), 2.0)+4.0*lambda*lambda*pos(0)*pos(0)); 
        }

        //evaluate the value of energy level chi_1 at position pos 
        real_t evaluate_pes_chi1(const RMatrixD1& pos) const
        {
            return 0.5*(omega.array()*pos.array()*pos.array()+(kappa1.array()+kappa2.array())*pos.array()).sum()+0.5*std::sqrt(std::pow(2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum(), 2.0)+4.0*lambda*lambda*pos(0)*pos(0)); 
        }

        //evaluate the gradient of energy level chi_0 at position pos
        RMatrixD1 evaluate_grad_chi0(const RMatrixD1& pos) const 
        {

            RMatrixD1  grad(D,1);
            //  \sqrt((2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m)^2+4\lambda^2Q_{10a}^2)
            real_t  sqrt_Q=std::sqrt(std::pow(2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum(), 2.0)+4.0*lambda*lambda*pos(0)*pos(0));

            //2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m
            real_t  Delta2Q=2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum();

            for(dim_t ii=0; ii<D; ii++){
            	grad(ii)=omega(ii)*pos(ii)+(kappa1(ii)+kappa2(ii))-0.25/sqrt_Q*(2.0*Delta2Q*(kappa2(ii)-kappa1(ii))+8.0*lambda*lambda*pos(0)*equalTo(ii,0));
            }
     		
            return grad;
        }

        //evaluate the gradient of energy level chi_1 at position pos
        RMatrixD1 evaluate_grad_chi1(const RMatrixD1& pos) const 
        {

            RMatrixD1  grad(D,1);

            //  \sqrt((2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m)^2+4\lambda^2Q_{10a}^2)
            real_t  sqrt_Q=std::sqrt(std::pow(2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum(), 2.0)+4.0*lambda*lambda*pos(0)*pos(0));

            //2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m
            real_t  Delta2Q=2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum();

            for(dim_t ii=0; ii<D; ii++){
            	grad(ii)=omega(ii)*pos(ii)+(kappa1(ii)+kappa2(ii))+0.25/sqrt_Q*(2.0*Delta2Q*(kappa2(ii)-kappa1(ii))+8.0*lambda*lambda*pos(0)*equalTo(ii,0));
            }
     		
            return grad;         
       
        }

        //evaluate the hessian of energy level chi_0 at position pos 
        RMatrixDD evaluate_hess_chi0(const RMatrixD1& pos) const 
        {
            RMatrixDD hess=RMatrixDD::Constant(0.0);


            //  \sqrt((2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m)^2+4\lambda^2Q_{10a}^2)
            real_t  sqrt_Q=std::sqrt(std::pow(2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum(), 2.0)+4.0*lambda*lambda*pos(0)*pos(0));

            //2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m
            real_t  Delta2Q=2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum();

            for(dim_t kk=0; kk<D; kk++){
            	for(dim_t ll=0; ll<D; ll++){
            		hess(kk,ll)=omega(kk)*equalTo(kk,ll)+0.125/std::pow(sqrt_Q, 3.0)*(2.0*Delta2Q*(kappa2(ll)-kappa1(ll))+8.0*lambda*lambda*pos(0)*equalTo(ll,0))*(2.0*Delta2Q*(kappa2(kk)-kappa1(kk))+8.0*lambda*lambda*pos(0)*equalTo(kk,0))-0.25/sqrt_Q*(2.0*(kappa2(ll)-kappa1(ll))*(kappa2(kk)-kappa1(kk))+8.0*lambda*lambda*equalTo(ll,0)*equalTo(kk,0));
            	}
            }


            return hess;

        }

        //evaluate the hessian of energy level chi_1 at position pos 
        RMatrixDD evaluate_hess_chi1(const RMatrixD1& pos) const 
        {
  
            RMatrixDD hess=RMatrixDD::Constant(0.0);


            //  \sqrt((2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m)^2+4\lambda^2Q_{10a}^2)
            real_t  sqrt_Q=std::sqrt(std::pow(2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum(), 2.0)+4.0*lambda*lambda*pos(0)*pos(0));

            //2\Delta+\sum_m(\kappa_m^2-\kappa_m^1)Q_m
            real_t  Delta2Q=2.0*Delta+((kappa2.array()-kappa1.array())*pos.array()).sum();

            for(dim_t kk=0; kk<D; kk++){
            	for(dim_t ll=0; ll<D; ll++){
            		hess(kk,ll)=omega(kk)*equalTo(kk,ll)-0.125/std::pow(sqrt_Q, 3.0)*(2.0*Delta2Q*(kappa2(ll)-kappa1(ll))+8.0*lambda*lambda*pos(0)*equalTo(ll,0))*(2.0*Delta2Q*(kappa2(kk)-kappa1(kk))+8.0*lambda*lambda*pos(0)*equalTo(kk,0))+0.25/sqrt_Q*(2.0*(kappa2(ll)-kappa1(ll))*(kappa2(kk)-kappa1(kk))+8.0*lambda*lambda*equalTo(ll,0)*equalTo(kk,0));
            	}
            }


            return hess;        
            

        }

        //evaluate the value of V_{00} at the qudrature node: nodes 
        CMatrix1X evaluate_pes_node_00(const CMatrixDX& nodes) const {

            dim_t n_nodes=nodes.cols();

            CMatrix1X pes(1, n_nodes);

            for(int ii=0; ii<n_nodes; ii++){
                pes(ii)=evaluate_pes_c00(nodes.col(ii));
            }
            return pes;
        }

        //evaluate the value of V_{11} at the qudrature node: nodes 
        CMatrix1X evaluate_pes_node_11(const CMatrixDX& nodes) const {

            dim_t n_nodes=nodes.cols();

            CMatrix1X pes(1, n_nodes);

            for(int ii=0; ii<n_nodes; ii++){
                pes(ii)=evaluate_pes_c11(nodes.col(ii));
            }
            return pes;
        }

        //evaluate the local quadratic of the energy level chi_0
        CMatrix1X local_quadratic_00(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            dim_t n_nodes=nodes.cols();
            real_t pes=evaluate_pes_chi0(pos);        //calculate the potential energy at position pos 
            RMatrixD1 grad=evaluate_grad_chi0(pos);   //calculate the gradient at position pos
            RMatrixDD hess=evaluate_hess_chi0(pos);   //calculate the hessian at position pos 

            CMatrix1X Vq=RMatrix1X::Constant(1, n_nodes, pes).template cast<complex_t>();  


            CMatrixDX dx = nodes.colwise() - pos.template cast<complex_t>();   // (x-q)

            CMatrix1X grad_dx=grad.transpose()*dx;                             //\Lambda{V}(q)(x-q)

            CMatrix1X hess_dx=(0.5*dx.transpose()*hess*dx).diagonal();         ///frac(1){2}(x-q)^T\Lambda^2V(q)(x-q)

            return Vq+grad_dx+hess_dx;                                         //return the local quadratic term 
        }


        //eevaluate the local quadratic of the energy level chi_1
        CMatrix1X local_quadratic_11(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            dim_t n_nodes=nodes.cols();
            real_t pes=evaluate_pes_chi1(pos);        //calculate the potential energy at position pos 
            RMatrixD1 grad=evaluate_grad_chi1(pos);   //calculate the gradient at position pos
            RMatrixDD hess=evaluate_hess_chi1(pos);   //calculate the hessian at position pos 

            CMatrix1X Vq=RMatrix1X::Constant(1, n_nodes, pes).template cast<complex_t>();  


            CMatrixDX dx = nodes.colwise() - pos.template cast<complex_t>();   // (x-q)

            CMatrix1X grad_dx=grad.transpose()*dx;                             //\Lambda{V}(q)(x-q)

            CMatrix1X hess_dx=(0.5*dx.transpose()*hess*dx).diagonal();         ///frac(1){2}(x-q)^T\Lambda^2V(q)(x-q)

            return Vq+grad_dx+hess_dx;                                         //return the local quadratic term 
        }


        //evaluate the local remainder of V_{00}
        CMatrix1X local_remainder_00(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            return evaluate_pes_node_00(nodes)-local_quadratic_00(nodes, pos);
        }

        //evaluate the local remainder of V_{11}
        CMatrix1X local_remainder_11(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            return evaluate_pes_node_11(nodes)-local_quadratic_11(nodes, pos);
        }

        //evaluate the local remainder of V_{00} for the homogenous Hagedorn wavepacket 
        CMatrix1X local_remainder_00_homogenous(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            if(leading_order_==0){
                
                return evaluate_pes_node_00(nodes)-local_quadratic_00(nodes, pos);
            }else{

                return evaluate_pes_node_00(nodes)-local_quadratic_11(nodes, pos);
            }

            
        }

        //evaluate the local remainder of V_{11} for the homogenous Hagedorn wavepacket 
        CMatrix1X local_remainder_11_homogenous(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            if(leading_order_==0){

                return evaluate_pes_node_11(nodes)-local_quadratic_00(nodes, pos);
            }else{

                return evaluate_pes_node_11(nodes)-local_quadratic_11(nodes, pos);
            }

        }

    };

}

