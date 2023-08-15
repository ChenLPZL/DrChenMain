#pragma once
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"


namespace potentials
{
    /**
    * \brief This class represents a simple avoided crossing: delta_gap potential 
    *
    * \tparam D dimensionality of 2D delta_gap (number of variables)
    *
    * \tparam N energy level of the 2D delta_gap (number of the PES)
    */

    template<dim_t D, dim_t N>
    struct delta_gap_2D_2N
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

        delta_gap_2D_2N()= default;  // default constructor 

        delta_gap_2D_2N(dim_t leading_order, real_t delta, const RMatrixDD &mass_inv): leading_order_(leading_order), delta_(delta), mass_inv_(mass_inv) {}  //constructor 

        //copy constructor 
        delta_gap_2D_2N(const delta_gap_2D_2N &that){

            leading_order_=that.leading_order_;
            delta_=that.delta_;
            mass_inv_=that.mass_inv_;
        }

        //assignment 
        delta_gap_2D_2N &operator=(const delta_gap_2D_2N& that){

            leading_order_=that.leading_order_;
            delta_=that.delta_;
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

        //evaluate the value of V_{ii}
        real_t evaluate_pes(const RMatrixD1& pos, dim_t ii) const 
        {
            
            if(ii==0){
                return evaluate_pes_00(pos);
            }else{
                return evaluate_pes_11(pos);
            }
        }

        //evaluate the value of V_{leading_order,leading_order}
        real_t evaluate_pes(const RMatrixD1& pos) const 
        {
            
            if(leading_order_==0){
                return evaluate_pes_00(pos);
            }else{
                return evaluate_pes_11(pos);
            }
        }


        //evaluate the gradient of V_{ii}
        RMatrixD1 evaluate_grad(const RMatrixD1& pos, dim_t ii) const 
        {

            if(ii==0){
                return evaluate_grad_00(pos);
            }else{
                return evaluate_grad_11(pos);
            }
        }

        //evaluate the gradient of V_{leading_order, leading_order}
        RMatrixD1 evaluate_grad(const RMatrixD1& pos) const 
        {

            if(leading_order_==0){
                return evaluate_grad_00(pos);
            }else{
                return evaluate_grad_11(pos);
            }
        }


        //evaluate the hessian of V_{ii}
        RMatrixDD evaluate_hess(const RMatrixD1& pos, dim_t ii) const 
        {
            
            if(ii==0){
                return evaluate_hess_00(pos);
            }else{
                return evaluate_hess_11(pos);
            }

        }

        //evaluate the hessian of V_{leading_order, leading_order}
        RMatrixDD evaluate_hess(const RMatrixD1& pos) const 
        {
            
            if(leading_order_==0){
                return evaluate_hess_00(pos);
            }else{
                return evaluate_hess_11(pos);
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

        // return the value of delta_
        real_t & delta()
        {

            return delta_;
        }

        real_t const& delta() const
        {

            return delta_;
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
        real_t   delta_;              // half of the energy level gap 
        RMatrixDD mass_inv_;          // inverse of the mass matrix M

        //evaluate the value of V_{00} at position pos
        complex_t evaluate_pes_c00(const CMatrixD1 &pos) const{

            return complex_t(0.5*std::tanh(std::sqrt(pos(0).real()*pos(0).real()+pos(1).real()*pos(1).real())), 0.0);
        }

        //evaluate the value of V_{01} at position pos
        complex_t evaluate_pes_c01(const CMatrixD1 &pos) const{

            return complex_t(delta_, 0.0);
        }

        //evaluate the value of V_{10} at position pos 
        complex_t evaluate_pes_c10(const CMatrixD1 &pos) const{

            return complex_t(delta_, 0.0);
        }

        //evaluate the value of V_{11} at position pos
        complex_t evaluate_pes_c11(const CMatrixD1 &pos) const{

            return complex_t(-0.5*std::tanh(std::sqrt(pos(0).real()*pos(0).real()+pos(1).real()*pos(1).real())), 0.0);
        }


        //evaluate the value of V_{00} at position pos 
        real_t evaluate_pes_00(const RMatrixD1& pos) const 
        {
            return 0.5*std::tanh(std::sqrt(pos(0)*pos(0)+pos(1)*pos(1)));
        }

        //evaluate the value of V_{11} at position pos 
        real_t evaluate_pes_11(const RMatrixD1& pos) const
        {
            return -0.5*std::tanh(std::sqrt(pos(0)*pos(0)+pos(1)*pos(1)));
        }

        //evaluate the gradient of V_{00} at position pos
        RMatrixD1 evaluate_grad_00(const RMatrixD1& pos) const 
        {

            RMatrixD1  grad(D,1);
            double x2y2_sqrt=std::sqrt(pos(0)*pos(0)+pos(1)*pos(1));   //sqrt(x^2+y^2)

            grad(0)=0.5/pow(std::cosh(x2y2_sqrt),2.0)/x2y2_sqrt*pos(0);
            grad(1)=0.5/pow(std::cosh(x2y2_sqrt),2.0)/x2y2_sqrt*pos(1);

            return grad;
        }

        //evaluate the gradient of V_{11} at position pos
        RMatrixD1 evaluate_grad_11(const RMatrixD1& pos) const 
        {

            return -1.0*evaluate_grad_00(pos);
        }

        //evaluate the hessian of V_{00} at position pos 
        RMatrixDD evaluate_hess_00(const RMatrixD1& pos) const 
        {
            RMatrixDD hess=RMatrixDD::Constant(0.0);

            double x2y2_sqrt=std::sqrt(pos(0)*pos(0)+pos(1)*pos(1));   //sqrt(x^2+y^2)

            hess(0,0)=-1.0/std::pow(1.0+std::cosh(2.0*x2y2_sqrt), 2.0)*std::sinh(2.0*x2y2_sqrt)*(2.0*pos(0)*pos(0))/std::pow(x2y2_sqrt, 2.0)-1.0/(1.0+std::cosh(2.0*x2y2_sqrt))*(pos(0)*pos(0))/std::pow(x2y2_sqrt,3.0)+1.0/(1.0+std::cosh(2.0*x2y2_sqrt))/x2y2_sqrt;
            hess(0,1)=-1.0/std::pow(1.0+std::cosh(2.0*x2y2_sqrt), 2.0)*std::sinh(2.0*x2y2_sqrt)*(2.0*pos(0)*pos(1))/std::pow(x2y2_sqrt, 2.0)-1.0/(1.0+std::cosh(2.0*x2y2_sqrt))*(pos(0)*pos(1))/std::pow(x2y2_sqrt,3.0);
            hess(1,0)=-1.0/std::pow(1.0+std::cosh(2.0*x2y2_sqrt), 2.0)*std::sinh(2.0*x2y2_sqrt)*(2.0*pos(0)*pos(1))/std::pow(x2y2_sqrt, 2.0)-1.0/(1.0+std::cosh(2.0*x2y2_sqrt))*(pos(0)*pos(1))/std::pow(x2y2_sqrt,3.0);
            hess(1,1)=-1.0/std::pow(1.0+std::cosh(2.0*x2y2_sqrt), 2.0)*std::sinh(2.0*x2y2_sqrt)*(2.0*pos(1)*pos(1))/std::pow(x2y2_sqrt, 2.0)-1.0/(1.0+std::cosh(2.0*x2y2_sqrt))*(pos(1)*pos(1))/std::pow(x2y2_sqrt,3.0)+1.0/(1.0+std::cosh(2.0*x2y2_sqrt))/x2y2_sqrt;

            return hess;

        }

        //evaluate the hessian of V_{11} at position pos 
        RMatrixDD evaluate_hess_11(const RMatrixD1& pos) const 
        {
          
            return -1.0*evaluate_hess_00(pos);

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

        //evaluate the local quadratic of the V_{00} 
        CMatrix1X local_quadratic_00(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            dim_t n_nodes=nodes.cols();
            real_t pes=evaluate_pes_00(pos);        //calculate the potential energy at position pos 
            RMatrixD1 grad=evaluate_grad_00(pos);   //calculate the gradient at position pos
            RMatrixDD hess=evaluate_hess_00(pos);   //calculate the hessian at position pos 

            CMatrix1X Vq=RMatrix1X::Constant(1, n_nodes, pes).template cast<complex_t>();  


            CMatrixDX dx = nodes.colwise() - pos.template cast<complex_t>();   // (x-q)

            CMatrix1X grad_dx=grad.transpose()*dx;                             //\Lambda{V}(q)(x-q)

            CMatrix1X hess_dx=(0.5*dx.transpose()*hess*dx).diagonal();         ///frac(1){2}(x-q)^T\Lambda^2V(q)(x-q)

            return Vq+grad_dx+hess_dx;                                         //return the local quadratic term 
        }


        //eevaluate the local quadratic of the V_{11} 
        CMatrix1X local_quadratic_11(const CMatrixDX& nodes, const RMatrixD1& pos) const {

            dim_t n_nodes=nodes.cols();
            real_t pes=evaluate_pes_11(pos);        //calculate the potential energy at position pos 
            RMatrixD1 grad=evaluate_grad_11(pos);   //calculate the gradient at position pos
            RMatrixDD hess=evaluate_hess_11(pos);   //calculate the hessian at position pos 

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

