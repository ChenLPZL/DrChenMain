#pragma once

#include "hawp_commons.hpp"
#include "innerproducts/homogeneous_inner_product.hpp"
#include "innerproducts/inhomogeneous_inner_product.hpp"
#include "innerproducts/vector_inner_product.hpp"

namespace observables{

    using innerproducts::HomogeneousInnerProduct;      // used for the energy, norm calculation 
    using innerproducts::InhomogeneousInnerProduct;    // used for the autocorrelation function 
    using innerproducts::VectorInnerProduct;           // used for the vectorized Hagedorn wavepacket  

    /**
    * \brief Compute the expectation value of the scalar potential V(x)
    *
    * \param packet   the scalar Hagedorn wavepacket 
    * \param V        the scalar potential V
    *
    * \tparam D        the dimension of the system 
    * \tparam ScalarPacket    the type for the scalar hagedorn wavepackeet 
    * \tparam QR       the quadrature rule used to calculate the innerproduct 
    * \tparam PES      the type for the scalar potential V
    */
    template<dim_t D, class ScalarPacket, class QR, class PES>
    real_t potential_energy(const ScalarPacket& packet, const PES& V){

        HomogeneousInnerProduct<D, QR> ip;
        return ip.quadrature(packet,
                                 [&V] (const CMatrix<D,Eigen::Dynamic>& nodes,
                                       const RMatrix<D,1>& pos)
                                 -> CMatrix<1,Eigen::Dynamic> {
                                    return V.pes().evaluate_pes_node(nodes);
                                 }).real();

    }

    /**
    * \brief Computes kinetic energy of a Hagedorn Wavepacket.
    *
    * \param packet       the scalar Hagedorn wavepacket 
    * \param mass_inv     the inverse of mass matrix 

    * \tparam D           Dimension of argument space
    * \tparam ScalarPacket     the type for the scalar Hagedorn wavepacket 
    */
    template<int D, class ScalarPacket>
    real_t kinetic_energy(const ScalarPacket& packet, const RMatrix<D, D>& mass_inv){

        CMatrix<D, Eigen::Dynamic> cprime=packet.apply_gradient();  //get the new expansion coefficient 

        complex_t result(0,0);
        for(dim_t ii=0; ii<D; ii++){
            result+=mass_inv(ii,ii)*cprime.row(ii).dot(cprime.row(ii));
        }

        return 0.5*result.real();     

    }


    /**
    * \brief Compute the norm of the Hagedorn wavepacket 
    *
    * \param packet   the scalar Hagedorn wavepacket 
    *
    * \tparam D        the dimension of the system 
    * \tparam ScalarPacket    the type of the scalar hagedorn wavepacket 
    */

    template<dim_t D, class ScalarPacket>
    real_t norm(const ScalarPacket& packet){

        return packet.coefficients().norm();

    }

    /**
    * \brief Compute the auto correlation of the two hagedorn wavepacket 
    *
    * \param pacbra   the bra Hagedorn wavepacket 
    * \param packet   the ket Hagedorn wavepacket 
    *
    * \tparam D        the dimension of the system 
    * \tparam ScalarPacbra    the type of the bra of the scalar hagedorn wavepacket 
    * \tparam ScalarPacket    the type of the ket of the scalar hagedorn wavepacket 
    * \tparam QR       the quadrature rule used to calculate the innerproduct 
    */

    template<dim_t D, class ScalarPacbra, class ScalarPacket, class QR>
    complex_t auto_corr(const ScalarPacbra& pacbra, const ScalarPacket& packet){

        InhomogeneousInnerProduct<D, QR> ip;
        return ip.quadrature(pacbra, packet);
    }

    /**
    * \brief Compute the potential energy of the vectorized Hagedorn wavepackets
    *
    * \param packet   the vectorized Hagedorn wavepacket 
    * \param V        the matrix potential V
    *
    * \tparam D        the dimension of the system 
    * \tparam VectorPacket    the type for the vectorized hagedorn wavepackeet 
    * \tparam QR       the quadrature rule used to calculate the innerproduct 
    * \tparam PES      the type for the matrix potential V
    */
    template<dim_t D, class VectorPacket, class QR, class PES>
    real_t potential_energy_vec(const VectorPacket& packet, const PES& V){

        VectorInnerProduct<D, QR> ip;

        return ip.quadrature(packet,
                                 [&V] (const CMatrix<D,Eigen::Dynamic>& nodes,
                                       const RMatrix<D,1>& pos, dim_t ii, dim_t jj)
                                 -> CMatrix<1,Eigen::Dynamic> {
                                    return V.pes().evaluate_pes_node(nodes, ii, jj);
                                 }).sum().real();
    }


    /**
    * \brief Computes kinetic energy of a vectorized Hagedorn Wavepacket.
    *
    * \param packet       the vectorized Hagedorn wavepacket 
    * \param mass_inv     the inverse of mass matrix 

    * \tparam D           Dimension of argument space
    * \tparam VectorPacket     the type for the vectorized Hagedorn wavepacket 
    */

    template<dim_t D, class VectorPacket>
    real_t kinetic_energy_vec(const VectorPacket& packet, const RMatrix<D, D>& mass_inv){
        
        const dim_t n_comps = packet.n_components();  //number of component of the hagedorn wavepacket 

        RMatrix<Eigen::Dynamic, 1> kin(n_comps, 1);  
        for(dim_t ii=0; ii<n_comps; ii++){
            kin(ii)=kinetic_energy(packet.component(ii), mass_inv);
        }

        return kin.sum();
    }

    /**
    * \brief Compute the norm of a vectorized Hagedorn wavepacket 
    *
    * \param packet   the vectorized Hagedorn wavepacket 
    *
    * \tparam D        the dimension of the system 
    * \tparam VectorPacket    the type of the vectorized hagedorn wavepacket 
    */

    template<dim_t D, class VectorPacket>
    real_t norm_vec(const VectorPacket& packet){

        /*
        VectorInnerProduct<D, QR> ip;
        return std::sqrt(ip.quadrature(packet).sum().real());
        */

        const dim_t n_comps= packet.n_components();  //number of component of the hagedorn wavepacket 

        RMatrix<Eigen::Dynamic, 1> pop_components(n_comps, 1); 
        for(dim_t ii=0; ii<n_comps; ii++){
            pop_components(ii)=packet.component(ii).coefficients().squaredNorm();
        }
        return std::sqrt(pop_components.sum());
    }

    /**
    * \brief Compute the population of a vectorized Hagedorn wavepacket 
    *
    * \param packet   the vectorized Hagedorn wavepacket 
    *
    * \tparam D        the dimension of the system 
    * \tparam VectorPacket    the type of the vectorized hagedorn wavepacket 
    */

    template<dim_t D, class VectorPacket>
    RMatrix<Eigen::Dynamic, 1> pop_vec(const VectorPacket& packet){

        const dim_t n_comps = packet.n_components();  //number of component of the hagedorn wavepacket 

        RMatrix<Eigen::Dynamic, 1> pop_components(n_comps, 1); 
        for(dim_t ii=0; ii<n_comps; ii++){
            pop_components(ii)=packet.component(ii).coefficients().squaredNorm();
        }

        return pop_components;
    }


    /**
    * \brief Compute the auto correlation of the two vectorized hagedorn wavepacket 
    *
    * \param pacbra   the bra vectorized Hagedorn wavepacket 
    * \param packet   the ket vectorized Hagedorn wavepacket 
    *
    * \tparam D        the dimension of the system 
    * \tparam VectorPacbra    the type of the bra of the vectorized hagedorn wavepacket 
    * \tparam VectorPacket    the type of the ket of the vectorized hagedorn wavepacket 
    * \tparam QR       the quadrature rule used to calculate the innerproduct 
    */

    template<dim_t D, class VectorPacbra, class VectorPacket, class QR>
    complex_t auto_corr_vec(const VectorPacbra& pacbra, const VectorPacket& packet){

        /*
        VectorInnerProduct<D, QR> ip;
        return ip.quadrature_inhomog(pacbra, packet).sum();
        */

        InhomogeneousInnerProduct<D, QR> ip;
        const dim_t n_comps = packet.n_components();  //number of component of the hagedorn wavepacket 
        CMatrix<Eigen::Dynamic, 1> auto_components(n_comps, 1); 
        for(dim_t ii=0; ii<n_comps; ii++){
            auto_components(ii)=ip.quadrature(pacbra.component(ii), packet.component(ii));
        }
        return auto_components.sum();

    }


}

