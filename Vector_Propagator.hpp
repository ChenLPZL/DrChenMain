#pragma once

#include <tuple>
#include "hawp_commons.hpp"
#include "innerproducts/vector_inner_product.hpp"

#include "potentials/potentials.hpp"


namespace propagators{

    template<dim_t D, class VectorPacket, class QR, class PES>
    class VectorHaWp_Propagator{
    public:

        using IP=innerproducts::VectorInnerProduct<D, QR>;          //the VectorInnerProduct type 
        using MatrixPES=potentials::MatrixPotentialMS<D, PES>;      //the matrix potential type 

        using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
        using RMatrixD1 = RMatrix<D, 1>;
        using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

        VectorHaWp_Propagator()=default;    //default constructor 


        //Constructor 
        VectorHaWp_Propagator(const MatrixPES& matrixms, const VectorPacket& packet)
        {
            matrixms_=matrixms;
            packet_=packet;
            mass_inv_=matrixms.pes().mass_inv();
        }

        //copy constructor 
        VectorHaWp_Propagator(const VectorHaWp_Propagator& that){ 
            
            matrixms_=that.matrixms_;
            packet_=that.packet_;
            mass_inv_=that.mass_inv_;
        }
    
        VectorHaWp_Propagator &operator=(const VectorHaWp_Propagator& that){  //assignment operator
            
            matrixms_=that.matrixms_;
            packet_=that.packet_;
            mass_inv_=that.mass_inv_;
            return *this;
        } 

        /**
        * \brief propagate homogenous hagedorn wavepacket based on the extended Lubich time-stepping algorithm (TVT second order)
        */
        void propagate_homo(real_t dt)
        {

            // Do a kinetic step of dt/2
            propagate_T(dt/2.0);

            // De a potential step of dt 
            propagate_V_homo(dt);

            //Do a kinetic step of dt/2
            propagate_T(dt/2.0);

        }

        /**
        * \brief propagate inhomogenous hagedorn wavepacket based on the extended Lubich time-stepping algorithm (TVT second order) 
        */
        void propagate_inhomo(real_t dt)
        {

            // Do a kinetic step of dt/2
            propagate_T(dt/2.0);

            // De a potential step of dt 
            propagate_V_inhomo(dt);

            //Do a kinetic step of dt/2
            propagate_T(dt/2.0);        

        }


        // fourth order propagation for homogenous wavepacket 
        void propagate_homo_order4(real_t dt)
        {
            double fac1=std::pow(2.0, 1.0/3.0);
            double fac2=1.0/(2.0-fac1);
            double fac_dt =fac2 * dt;
            double coc1 = fac_dt / 2;
            double coc2 = fac_dt * (1 - fac1) / 2;
            double cod1 = fac_dt;
            double cod2 = -fac1 * fac_dt;

            propagate_T(coc1);
            propagate_V_homo(cod1);
            propagate_T(coc2);
            propagate_V_homo(cod2);
            propagate_T(coc2);
            propagate_V_homo(cod1);
            propagate_T(coc1);
        }

        //fourth order propagation for inhomogenous wavepacket 
        void propagate_inhomo_order4(real_t dt)
        {
            double fac1=std::pow(2.0, 1.0/3.0);
            double fac2=1.0/(2.0-fac1);
            double fac_dt =fac2 * dt;
            double coc1 = fac_dt / 2;
            double coc2 = fac_dt * (1 - fac1) / 2;
            double cod1 = fac_dt;
            double cod2 = -fac1 * fac_dt;

            propagate_T(coc1);
            propagate_V_inhomo(cod1);
            propagate_T(coc2);
            propagate_V_inhomo(cod2);
            propagate_T(coc2);
            propagate_V_inhomo(cod1);
            propagate_T(coc1);
        }      


        /**
        * \brief Grants access to the matrix potential 
        */

        MatrixPES & matrixms()
        {
            return matrixms_;
        }

        MatrixPES const& matrixms() const
        {
            return matrixms_;
        }
            
        /**
        * \brief Grants access to the vectorized Hagedorn wavepacket 
        */
        VectorPacket & packet()
        {
            return packet_;
        }

        VectorPacket const& packet() const
        {
            return packet_;
        }
    

    private:
        MatrixPES matrixms_;           //the potential wrapper for any matrix potential 
        VectorPacket    packet_;       //the scalar hagedorn wavepacket 
        RMatrix<D, D>  mass_inv_;      //the inverse mass matrix 

        //propagate kinetic operator for both homogenous and inhomogenous wavepacket 
        void propagate_T(real_t dt)
        {
            const dim_t NN=packet_.n_components();     // number of component 
            
            for(dim_t ii=0; ii<NN; ii++){

                packet_.component(ii).parameters().updateq( dt*mass_inv_*packet_.component(ii).parameters().p() );  
                packet_.component(ii).parameters().updateQ( dt*mass_inv_*packet_.component(ii).parameters().P() );  
                packet_.component(ii).parameters().updateS( complex_t(0.5*dt*packet_.component(ii).parameters().p().dot(mass_inv_*packet_.component(ii).parameters().p()),0.0) ); 
            }

        }

        //propagate potential operator for homogenous wavepacket 
        void propagate_V_homo(real_t dt)
        {

            const dim_t NN=packet_.n_components();     // number of component 

            // Calculate offsets into output matrix.
            std::vector<dim_t> offsets=packet_.offset();

            // the number of coefficients for all component of the Hagedorn wavepacket.
            std::size_t total_size = packet_.size();

            //calculate local quadratic 
            for(dim_t ii=0; ii<NN; ii++){

                packet_.component(ii).parameters().updatep( -dt*matrixms_.pes().evaluate_grad(packet_.component(ii).parameters().q()) ); //p=p-dt\Lambda{V}_{xx}(q)
                packet_.component(ii).parameters().updateP( -dt*matrixms_.pes().evaluate_hess(packet_.component(ii).parameters().q())*packet_.component(ii).parameters().Q() ); //P=P-dt\Lambda^2{V}_{xx}(q)Q
                packet_.component(ii).parameters().updateS( complex_t(-dt*matrixms_.pes().evaluate_pes(packet_.component(ii).parameters().q()),0.0) ); //S=S-dt{V}_{xx}(q)

            }

            //update the coefficient of the hagedorn wavepacket, calculate the local remainder 
            auto fun =
            [this] (CMatrixDX nodes, RMatrixD1 pos, dim_t ii, dim_t jj)  -> CMatrix1X {
                    return matrixms_.pes().local_remainder_homogenous(nodes, pos, ii, jj);
            };

            CMatrix<Eigen::Dynamic, Eigen::Dynamic> FF = IP::build_matrix(packet_, fun);   //calculate the block matrix F

            complex_t factor = -dt * complex_t(0,1) / (packet_.eps()*packet_.eps());

            //stack the coefficient vectors of all components 
            Coefficients coeffs(total_size, 1);   
            for (dim_t ii = 0; ii < NN; ++ii){

                coeffs.block(offsets[ii], 0, packet_.component(ii).coefficients().size(), 1)=packet_.component(ii).coefficients();
                
            } 

            //propagate the coefficients and split the coefficient 
            Coefficients coeffs_split(total_size, 1); 
            coeffs_split=(factor * FF).exp() * coeffs;  //c=exp(-dt\frac{i}{\epsilon^2}F)c
            for(dim_t ii=0; ii<NN; ++ii){
                packet_.component(ii).coefficients()=coeffs_split.block(offsets[ii], 0, packet_.component(ii).coefficients().size(), 1);
            }

        }

        //propagate potential operator for inhomogenous wavepacket 
        void propagate_V_inhomo(real_t dt)
        {

            const dim_t NN=packet_.n_components();     // number of component 

            // Calculate offsets into output matrix.
            std::vector<dim_t> offsets=packet_.offset();

            // the number of coefficients for all component of the Hagedorn wavepacket.
            std::size_t total_size = packet_.size();

            //calculate local quadratic 
            for(dim_t ii=0; ii<NN; ii++){

                packet_.component(ii).parameters().updatep( -dt*matrixms_.pes().evaluate_grad(packet_.component(ii).parameters().q(),ii) ); //p_i=p_i-dt\Lambda{V}_{ii}(q_i)
                packet_.component(ii).parameters().updateP( -dt*matrixms_.pes().evaluate_hess(packet_.component(ii).parameters().q(),ii)*packet_.component(ii).parameters().Q() ); //P_i=P_i-dt\Lambda^2{V}_{ii}(q_i)Q_i
                packet_.component(ii).parameters().updateS( complex_t(-dt*matrixms_.pes().evaluate_pes(packet_.component(ii).parameters().q(),ii),0.0) ); //S_i=S_i-dt{V}_{ii}(q_i)

            }

            //update the coefficient of the hagedorn wavepacket, calculate the local remainder 
            auto fun =
            [this] (CMatrixDX nodes, RMatrixD1 pos, dim_t ii, dim_t jj)  -> CMatrix1X {
                    return matrixms_.pes().local_remainder(nodes, pos, ii, jj);
            };

            CMatrix<Eigen::Dynamic, Eigen::Dynamic> FF = IP::build_matrix(packet_, fun);   //calculate the block matrix F

            complex_t factor = -dt * complex_t(0,1) / (packet_.eps()*packet_.eps());

            //stack the coefficient vectors of all components 
            Coefficients coeffs(total_size, 1);   
            for (dim_t ii = 0; ii < NN; ++ii){
                coeffs.block(offsets[ii], 0, packet_.component(ii).coefficients().size(), 1)=packet_.component(ii).coefficients();
            } 

            //propagate the coefficients and split the coefficient 
            Coefficients coeffs_split(total_size, 1); 
            coeffs_split=(factor * FF).exp() * coeffs;  //c=exp(-dt\frac{i}{\epsilon^2}F)c
            for(dim_t ii=0; ii<NN; ++ii){
                packet_.component(ii).coefficients()=coeffs_split.block(offsets[ii], 0, packet_.component(ii).coefficients().size(), 1);
            }


        }


    };

}

