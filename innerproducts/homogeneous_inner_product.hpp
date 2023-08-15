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
    * \brief Class providing homogeneous inner product calculation of scalar
    *   wavepackets.
    *
    * \tparam D dimensionality of processed wavepackets
    * \tparam QR quadrature rule to use, with R nodes
    */
    template<dim_t D, class QR>
    class HomogeneousInnerProduct
    {
    public:
        using CMatrixXX = CMatrix<Eigen::Dynamic, Eigen::Dynamic>;
        using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
        using CMatrixX1 = CMatrix<Eigen::Dynamic, 1>;
        using CMatrixD1 = CMatrix<D, 1>;
        using CMatrixDD = CMatrix<D, D>;
        using CMatrixDX = CMatrix<D, Eigen::Dynamic>;
        using RMatrixD1 = RMatrix<D, 1>;
        using CDiagonalXX = Eigen::DiagonalMatrix<complex_t, Eigen::Dynamic>;
        using NodeMatrix = typename QR::NodeMatrix;
        using WeightVector = typename QR::WeightVector;
        using op_t = std::function<CMatrix1X(CMatrixDX,RMatrixD1)>;

        /**
        * \brief Calculate the matrix of the inner product.
        *
        * Returns the matrix elements \f$\langle \Phi | f | \Phi \rangle\f$ with
        * an operator \f$f\f$.
        * The coefficients of the wavepacket are ignored.
        *
        * \param[in] packet wavepacket \f$\Phi\f$
        * \param[in] op operator \f$f(x, q) : \mathbb{C}^{D \times R} \times
        *   \mathbb{R}^D \rightarrow \mathbb{C}^R\f$ which is evaluated at the
        *   nodal points \f$x\f$ and position \f$q\f$;
        *   default returns a vector of ones
        */
        
        // Algorithm 11 Build the matrix F of matrix elements of f 
        template<class ScalarPacket>
        static CMatrixXX build_matrix(const ScalarPacket& packet,
                                          const op_t& op=default_op) {
            
            const dim_t n_nodes = QR::number_nodes();   // number of nodes 
            const CMatrixD1& q = packet.parameters().q().template cast<complex_t>();  // obtain q_0=q (see Eq (4.10) for homogenous case )
            const CMatrixDD& Q = packet.parameters().Q();   //  Obtain Q (see Eq. (4.11))
            NodeMatrix nodes;       //matrix nodes (size: D*|R|) where |R| is the number of nodes 
            WeightVector weights;   //weights (size: 1*|R|) where |R| is the number of nodes 
            std::tie(nodes, weights) = QR::nodes_and_weights(); //unpack the tuple into individual objects (nodes, weights)


            // Compute affine transformation
            const CMatrixDD Qs = (Q * Q.adjoint()).sqrt();  //obtain Q_0=(QQ^H)^{-1} (Eq. 4.11) and Q_S=(\sqrt{Q_0})^{-1} (Eq. 4.12)

            // Transform nodes
            const CMatrixDX transformed_nodes = q.replicate(1, n_nodes) + packet.eps() * (Qs * nodes);  //Eq (4.16) obtain the transformed nodes (D*|R|) where |R| is the number of nodes 


            // Apply operator
            const CMatrix1X values = op(transformed_nodes, packet.parameters().q());  //f(\gamma_r^{''}) (see Eq. 4.18)

            // Prefactor
            const CMatrix1X factor =
            // std::conj(packet.prefactor()) * packet.prefactor() * Qs.determinant() = 1
            std::pow(packet.eps(), D) * weights.array() * values.array();

            // Evaluate basis
            const CMatrixXX basis = packet.evaluate_basis(transformed_nodes);

            // Build matrix
            const CDiagonalXX Dfactor(factor);
            const CMatrixXX result = basis.matrix().conjugate() * Dfactor * basis.matrix().transpose();  // see Eq (4.18)

            // Global phase cancels out
            return result;
        }

        /**
        * \brief Perform quadrature.
        *
        * Evaluates the scalar \f$\langle \Phi | f | \Phi \rangle\f$.
        * See build_matrix() for the parameters.
        */
        template<class ScalarPacket>
        static complex_t quadrature(const ScalarPacket& packet,
                                        const op_t& op=default_op) {
            const auto M = build_matrix(packet, op);
            // Quadrature with wavepacket coefficients, c^H M c.
            return packet.coefficients().adjoint() * M * packet.coefficients();
        }

    private:
        static CMatrix1X default_op(const CMatrixDX& nodes, const RMatrixD1& pos)
        {
            (void)pos;
            return CMatrix1X::Ones(1, nodes.cols());
        }
    };
}
