#pragma once


#include "hawp_commons.hpp"
#include "innerproducts/homogeneous_inner_product.hpp"
#include "potentials/potentials.hpp"


namespace propagators{

    template<dim_t D, class ScalarPacket, class QR, class PES>
    class ScalarHaWp_Propagator{
    public:

        using IP = innerproducts::HomogeneousInnerProduct<D, QR>;   //the homogeneousInnerProduct type 
        using ScalarPES=potentials::MatrixPotential1S<D, PES>;             //the Scalar potential type 

        using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
        using RMatrixD1 = RMatrix<D, 1>;
        using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

        ScalarHaWp_Propagator() = default;  // default constructor 

        //Constructor 
        ScalarHaWp_Propagator(const ScalarPES& matrix1s, const ScalarPacket& packet)
        {
            matrix1s_=matrix1s;
            packet_=packet;
            mass_inv_=matrix1s.pes().mass_inv();
        }

        //copy constructor 
        ScalarHaWp_Propagator(const ScalarHaWp_Propagator& that){ 
            
            matrix1s_=that.matrix1s_;
            packet_=that.packet_;
            mass_inv_=that.mass_inv_;
        }
    
        ScalarHaWp_Propagator &operator=(const ScalarHaWp_Propagator& that){  //assignment operator
            
            matrix1s_=that.matrix1s_;
            packet_=that.packet_;
            mass_inv_=that.mass_inv_;
            return *this;
        } 


        /**
        * \brief propagate scalar hagedorn wavepacket based on Lubich's time stepping algorithm (second order TVT)
        */
        void propagate(real_t dt)
        {
            // Do a kinetic step of dt/2
            propagate_T(dt/2.0);

            // De a potential step of dt 
            propagate_V(dt);

            //Do a kinetic step of dt/2
            propagate_T(dt/2.0);

        }

        // fourth order propagation 
        void propagate_order4(real_t dt)
        {
            double fac1=std::pow(2.0, 1.0/3.0);
            double fac2=1.0/(2.0-fac1);
            double fac_dt =fac2 * dt;
            double coc1 = fac_dt / 2;
            double coc2 = fac_dt * (1 - fac1) / 2;
            double cod1 = fac_dt;
            double cod2 = -fac1 * fac_dt;

            propagate_T(coc1);
            propagate_V(cod1);
            propagate_T(coc2);
            propagate_V(cod2);
            propagate_T(coc2);
            propagate_V(cod1);
            propagate_T(coc1);
        }


        /**
        * \brief Grants access to the scalar potential V(x)
        */

        ScalarPES & matrix1s()
        {
            return matrix1s_;
        }

        ScalarPES const& matrix1s() const
        {
            return matrix1s_;
        }
            
        /**
        * \brief Grants access to the scalar hagedorn wavepacket 
        */
        ScalarPacket & packet()
        {
            return packet_;
        }

        ScalarPacket const& packet() const
        {
            return packet_;
        }

    private:
        ScalarPES matrix1s_;     //the potential wrapper for any scalar potential V(x)
        ScalarPacket    packet_;       //the scalar hagedorn wavepacket 
        RMatrix<D, D>  mass_inv_;  //the inverse mass matrix 

        //propagate the kinetic operator 
        void propagate_T(real_t dt){

            packet_.parameters().updateq( dt*mass_inv_*packet_.parameters().p() );   
            packet_.parameters().updateQ( dt*mass_inv_*packet_.parameters().P() );   
            packet_.parameters().updateS( complex_t(0.5*dt*packet_.parameters().p().dot(mass_inv_*packet_.parameters().p()),0.0) );            
        }

        //propagate the potential operator 
        void propagate_V(real_t dt){

            //calculate local quadratic 
            packet_.parameters().updatep( -dt*matrix1s_.pes().evaluate_grad(packet_.parameters().q()) );   
            packet_.parameters().updateP( -dt*matrix1s_.pes().evaluate_hess(packet_.parameters().q())*packet_.parameters().Q() );  
            packet_.parameters().updateS( complex_t(-dt*matrix1s_.pes().evaluate_pes(packet_.parameters().q()),0.0) );    

            //update the coefficient of the hagedorn wavepacket, calculate the local remainder    
            auto fun =
            [this] (CMatrixDX nodes, RMatrixD1 pos)  -> CMatrix1X {
                    return matrix1s_.pes().local_remainder(nodes, pos);
            };

          
            CMatrix<Eigen::Dynamic, Eigen::Dynamic> FF = IP::build_matrix(packet_, fun);  

            complex_t factor = -dt * complex_t(0,1) / (packet_.eps()*packet_.eps());
            Coefficients  coeffs=packet_.coefficients();    
            packet_.coefficients()=(factor * FF).exp() * coeffs; 

        }

    };

}

