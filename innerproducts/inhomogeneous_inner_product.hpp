#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "../types.hpp"
#include "../hawp_commons.hpp"


namespace innerproducts {

    /**
    * \brief Class providing inhomogeneous inner product calculation of scalar
    *   wavepackets.
    *
    * \tparam D dimensionality of processed wavepackets
    * \tparam QR quadrature rule to use, with R nodes
    */
    
    template<dim_t D, class QR>
    class InhomogeneousInnerProduct
    {
    public:
        using CMatrixXX = CMatrix<Eigen::Dynamic, Eigen::Dynamic>;
        using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
        using CMatrixX1 = CMatrix<Eigen::Dynamic, 1>;
        using CMatrixD1 = CMatrix<D, 1>;
        using CMatrixDD = CMatrix<D, D>;
        using CMatrixDX = CMatrix<D, Eigen::Dynamic>;
        using RMatrixDD = RMatrix<D, D>;
        using RMatrixD1 = RMatrix<D, 1>;
        using CDiagonalXX = Eigen::DiagonalMatrix<complex_t, Eigen::Dynamic>;
        using NodeMatrix = typename QR::NodeMatrix;
        using WeightVector = typename QR::WeightVector;
        using op_t = std::function<CMatrix1X(CMatrixDX,RMatrixD1)>;

        /**
        * \brief Calculate the matrix of the inner product.
        *
        * Returns the matrix elements \f$\langle \Phi | f | \Phi' \rangle\f$ with
        * an operator \f$f\f$.
        * The coefficients of the wavepackets are ignored.
        *
        * \param[in] pacbra wavepacket \f$\Phi\f$
        * \param[in] packet wavepacket \f$\Phi'\f$
        * \param[in] op operator \f$f(x, q) : \mathbb{C}^{D \times R} \times
        *   \mathbb{R}^D \rightarrow \mathbb{C}^R\f$ which is evaluated at the
        *   nodal points \f$x\f$ and position \f$q\f$;
        *   default returns a vector of ones
        */

        template<class ScalarPacbra, class ScalarPacket>
        static CMatrixXX build_matrix(const ScalarPacbra& pacbra,
                                          const ScalarPacket& packet,
                                          const op_t& op=default_op) {
            
            const dim_t n_nodes = QR::number_nodes();   // the number of nodes 
            const complex_t S_bra = pacbra.parameters().S();   // the phase S of bra 
            const complex_t S_ket = packet.parameters().S();   // the phase S of ket 
            NodeMatrix nodes;                                  //the node matrix (size: D* |R) where |R| is the number of nodes 
            WeightVector weights;
            std::tie(nodes, weights) = QR::nodes_and_weights(); //unpack the tuple into individual objects (nodes, weights)

            // Mix parameters and compute affine transformation
            std::pair<RMatrixD1, RMatrixDD> PImix = pacbra.parameters().mix(packet.parameters());  //get the mixing value of two sets II_r and II_c of Hagedorn parameters (see Algorithm 10)
            const RMatrixD1& q0 = std::get<0>(PImix);   //q_0 (see Eq. 4.16) 
            const RMatrixDD& Qs = std::get<1>(PImix);   //Q_S (see Eq. 4.16)

            // Transform nodes (see Eq. 4.16)
            const CMatrixDX transformed_nodes = q0.template cast<complex_t>().replicate(1, n_nodes) + packet.eps() * (Qs.template cast<complex_t>() * nodes);

            // Apply operator
            const CMatrix1X values = op(transformed_nodes, q0); //obtain f(\gamma_r^{'})

            // Prefactor (see Eq. 4. 17) please note that when we calculate the basis function value at nodes, we omit the prefector of 1/\sqrt{det{Q}}.
            const CMatrix1X factor =
                std::conj(pacbra.prefactor()) * packet.prefactor() * Qs.determinant() *
                std::pow(packet.eps(), D) * weights.array() * values.array();    

            // Evaluate basis
            const CMatrixXX basisr = pacbra.evaluate_basis(transformed_nodes);
            const CMatrixXX basisc = packet.evaluate_basis(transformed_nodes);

            // Build matrix
            const CDiagonalXX Dfactor(factor);
            const CMatrixXX result = basisr.matrix().conjugate() * Dfactor * basisc.matrix().transpose();

            // Global phase
            const complex_t phase = std::exp(complex_t(0,1) * (S_ket - std::conj(S_bra)) / std::pow(packet.eps(),2));
            return phase * result;
        }

        /**
        * \brief Perform quadrature.
        *
        * Evaluates the scalar \f$\langle \Phi | f | \Phi' \rangle\f$.
        * See build_matrix() for the parameters.
        */
        template<class ScalarPacbra, class ScalarPacket>
        static complex_t quadrature(const ScalarPacbra& pacbra,
                                        const ScalarPacket& packet,
                                        const op_t& op=default_op) {
            const auto M = build_matrix(pacbra, packet, op);
                // Quadrature with wavepacket coefficients, c^H M c.
            return pacbra.coefficients().adjoint() * M * packet.coefficients();
        }

    private:
        static CMatrix1X default_op(const CMatrixDX& nodes, const RMatrixD1& pos)
        {
            (void)pos;
            return CMatrix1X::Ones(1, nodes.cols());
        }
    };
}

