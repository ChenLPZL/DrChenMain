#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../types.hpp"
#include "../hawp_commons.hpp"

#include "inhomogeneous_inner_product.hpp"


namespace innerproducts {

	/**
    * \brief Class providing inner product calculation of multi-component
    *  wavepackets.
    *
    * \tparam D dimensionality of processed wavepackets
    * \tparam QR quadrature rule to use, with R nodes
    */
     
    template<dim_t D, class QR>
    class VectorInnerProduct
    {
    public:
        
        using CMatrixNN = CMatrix<Eigen::Dynamic, Eigen::Dynamic>;
        using CMatrix1N = CMatrix<1, Eigen::Dynamic>;
        using CMatrixN1 = CMatrix<Eigen::Dynamic, 1>;
        using CMatrixD1 = CMatrix<D, 1>;
        using CMatrixDD = CMatrix<D, D>;
        using CMatrixDN = CMatrix<D, Eigen::Dynamic>;
        using RMatrixD1 = RMatrix<D, 1>;
        using CDiagonalNN = Eigen::DiagonalMatrix<complex_t, Eigen::Dynamic>;
        using NodeMatrix = typename QR::NodeMatrix;
        using WeightVector = typename QR::WeightVector;
        using op_t = std::function<CMatrix1N(CMatrixDN,RMatrixD1, dim_t, dim_t)>;


        /**
        * \brief Calculate the matrix of the inner product.
        *
        * Returns the matrix elements \f$\langle \Psi | f | \Psi \rangle\f$ with
        * an operator \f$f\f$.
        * The matrix consists of \f$N \times N\f$ blocks (\f$N\f$: number of
        * components), each of size \f$|\mathfrak{K}| \times |\mathfrak{K}|\f$.
        * The coefficients of the wavepacket are ignored.
        *
        * \param[in] packet multi-component wavepacket \f$\Psi\f$
        * \param[in] op operator \f$f(x, q, i, j) : \mathbb{C}^{D \times R} \times
        * \mathbb{R}^D \times \mathbb{N} \times \mathbb{N} \rightarrow
        * \mathbb{C}^R\f$ which is evaluated at the
        * nodal points \f$x\f$ and position \f$q\f$, between components
        *  \f$\Phi_i\f$ and \f$\Phi_j\f$;
        *  default returns a vector of ones if i==j, zeros otherwise
        *
        * \tparam Packet packet type 
        */
        template<class Packet>
        static CMatrixNN build_matrix(const Packet& packet,
                                    const op_t& op=default_op) {
            
            const dim_t n_comps = packet.n_components();  //number of component of the hagedorn wavepacket 

            // Calculate offsets into output matrix.
            std::vector<dim_t> offsets=packet.offset();

            // Allocate output matrix.
            std::size_t total_size = packet.size();
            CMatrixNN result(total_size, total_size);

            // Calculate matrix.

            using IP=InhomogeneousInnerProduct<D, QR>;
            for (dim_t i = 0; i < n_comps; ++i) {
                for (dim_t j = 0; j < n_comps; ++j) {
                    using namespace std::placeholders;   //for _1, _2, _3...
                    result.block(offsets[i], offsets[j],
                                    packet.component(i).coefficients().size(),
                                    packet.component(j).coefficients().size()) =
                            IP::build_matrix(packet.component(i), packet.component(j),
                                             std::bind(op, _1, _2, i, j));
                }
            }
            return result;
        }

        /**
        * \brief Calculate the matrix of the inner product.
        *
        * Returns the matrix elements \f$\langle \Psi | f | \Psi' \rangle\f$ with
        * an operator \f$f\f$.
        * The matrix consists of \f$N \times N'\f$ blocks (\f$N,N'\f$: number of
        * components of \f$\Psi,\Psi'\f$), each of size \f$|\mathfrak{K}_i| \times
        * |\mathfrak{K}_j'|\f$.  The coefficients of the wavepacket are ignored.
        *
        * \param[in] pacbra multi-component wavepacket \f$\Psi\f$
        * \param[in] packet multi-component wavepacket \f$\Psi'\f$
        * \param[in] op operator \f$f(x, q, i, j) : \mathbb{C}^{D \times R} \times
        *   \mathbb{R}^D \times \mathbb{N} \times \mathbb{N} \rightarrow
        *   \mathbb{C}^R\f$ which is evaluated at the
        *   nodal points \f$x\f$ and position \f$q\f$, between components
        *   \f$\Phi_i\f$ and \f$\Phi_j\f$;
        *   default returns a vector of ones if i==j, zeros otherwise
        *
        * \tparam Pacbra packet type of \f$\Psi\f$ 
        * \tparam Packet packet type of \f$\Psi'\f$
        */
        template<class Pacbra, class Packet>
        static CMatrixNN build_matrix_inhomog(const Pacbra& pacbra,
                                                  const Packet& packet,
                                                  const op_t& op=default_op) {
            
            const dim_t n_components_bra = pacbra.n_components();
            const dim_t n_components_ket = packet.n_components();

            // Calculate offsets into output matrix.
            std::vector<dim_t> offsets_bra=pacbra.offset();
            std::vector<dim_t> offsets_ket=packet.offset();

            // Allocate output matrix.
            std::size_t total_rows=pacbra.size();
            std::size_t total_cols=packet.size();
            CMatrixNN result(total_rows, total_cols);

            // Calculate matrix.

            using IP=InhomogeneousInnerProduct<D, QR>; 
            for (dim_t i = 0; i < n_components_bra; ++i) {
                for (dim_t j = 0; j < n_components_ket; ++j) {
                    using namespace std::placeholders;    //for _1, _2, _3...
                    result.block(offsets_bra[i], offsets_ket[j],
                                     pacbra.component(i).coefficients().size(),
                                     packet.component(j).coefficients().size()) =
                            IP::build_matrix(pacbra.component(i), packet.component(j),
                                             std::bind(op, _1, _2, i, j));
                }
            }
            return result;
        }

        /**
        * \brief Perform quadrature.
        *
        * Returns an \f$N^2\f$-sized vector of scalars \f$\langle \Phi_i | f |
        * \Phi_j \rangle\f$.
        * See build_matrix() for the parameters.
        */
        template<class Packet>
        static CMatrixN1 quadrature(const Packet& packet,
                                        const op_t& op=default_op) {
            const dim_t n_comps = packet.n_components();
            CMatrixN1 result(n_comps * n_comps, 1);

            // Calculate matrix.

            using IP=InhomogeneousInnerProduct<D, QR>;
            for (dim_t i = 0; i < n_comps; ++i) {
                for (dim_t j = 0; j < n_comps; ++j) {
                    using namespace std::placeholders;
                    result(j + i * n_comps)=packet.component(i).coefficients().dot(IP::build_matrix(packet.component(i), packet.component(j),std::bind(op, _1, _2, i, j)) * packet.component(j).coefficients());
                }
            }
            return result;
        }

        /**
        * \brief Perform quadrature.
        *
        * Returns an \f$N \cdot N'\f$-sized vector of scalars \f$\langle \Phi_i | f | \Phi'_j \rangle\f$.
        * See build_matrix_inhomog() for the parameters.
        */
        template<class Pacbra, class Packet>
        static CMatrixN1 quadrature_inhomog(const Pacbra& pacbra,
                                                const Packet& packet,
                                                const op_t& op=default_op) {
            const dim_t n_components_bra = pacbra.n_components();
            const dim_t n_components_ket = packet.n_components();
            CMatrixN1 result(n_components_bra * n_components_ket, 1);

            // Calculate matrix.     

            using IP=InhomogeneousInnerProduct<D, QR>;
            for (dim_t i = 0; i < n_components_bra; ++i) {
                for (dim_t j = 0; j < n_components_ket; ++j) {
                    using namespace std::placeholders;

                    result(j + i * n_components_ket)= pacbra.component(i).coefficients().dot(IP::build_matrix(pacbra.component(i), packet.component(j), std::bind(op, _1, _2, i, j)) *packet.component(j).coefficients());
                    
                }
            }
            return result;
        }

    private:
        static CMatrix1N default_op(const CMatrixDN& nodes, const RMatrixD1& pos, dim_t i, dim_t j)
        {
            (void)pos;
            if (i == j) return CMatrix1N::Ones(1, nodes.cols());
            else        return CMatrix1N::Zero(1, nodes.cols());
        }
    };
        
}

