#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <Eigen/Core>

#include "hawp_paramset.hpp"
#include "multi_index.hpp"
#include "shape.hpp"
#include "types.hpp"
#include "kahan_sum.hpp"


namespace wavepackets {

    /**
    * \brief implementation of a scalar Hagedorn wavepacket 
    */
    template<dim_t D, class Shape>
    class ScalarHaWp
    {
    private:
        real_t eps_;     // the semiclassical scaling parameter 
        HaWpParamSet<D> parameters_;        //Hagedorn paramet set (II=p,q,P,Q,S)
        Coefficients coefficients_;         //Coefficient of the Hagedorn wavepackets (c_k)
        Shape        shape_;                //basis shape (see shape.hpp for various basis shapes)

        std::vector<real_t> sqrt_;         //lookup-table for sqrt 

    public:


    	ScalarHaWp() = default;  // default constructor 

    	//Constructor 
    	ScalarHaWp(real_t eps, const HaWpParamSet<D>& parameter, const Coefficients& coefficients, const Shape& shape)
    	{
    		eps_=eps;
    		parameters_=parameter;
    		coefficients_=coefficients;
    		shape_=shape;

    		int limit=-1;
    		for (dim_t d = 0; d < D; d++)
    			limit=std::max(limit, shape_.bbox(d));

    		sqrt_.resize(limit+1);
    		for (int i = 0; i <= limit; i++)
    			sqrt_[i] = std::sqrt( real_t(i) );
    	}

        //copy constructor 
        ScalarHaWp(const ScalarHaWp& that){ 
            
   			eps_=that.eps_;
   			parameters_=that.parameters_;
   			coefficients_=that.coefficients_;
   			shape_=that.shape_;
   			sqrt_=that.sqrt_;
        }
    
        ScalarHaWp &operator=(const ScalarHaWp& that){  //assignment operator
            
        	eps_=that.eps_;
        	parameters_=that.parameters_;
        	coefficients_=that.coefficients_;
        	shape_=that.shape_;
        	sqrt_=that.sqrt_;
            return *this;
        } 

        /**
        * \brief Evaluates all basis functions \f$ \{\phi_k\} \f$ on complex grid nodes \f$ x \in \gamma \f$.
        *
        * \param grid
        * Complex grid nodes / quadrature points \f$ \gamma \f$.
        * Complex matrix with shape (dimensionality, number of grid nodes).
        * \return Complex 2D-array with shape (basis shape size, number of grid nodes)
        * not including (detQ)^{-1/2}
        */
        template<int N> HaWpBasisVector<N>
        evaluate_basis(CMatrix<D,N> const& grid) const
        {
            RMatrix<D,1> const& q = parameters_.q();
            CMatrix<D,D> const& Q = parameters_.Q();
            RMatrix<D,1> const& p = parameters_.p();
            CMatrix<D,D> const& P = parameters_.P();

            // precompute ...
            CMatrix<D,N> dx_ = grid.colwise() - q.template cast<complex_t>();
            CMatrix<D,D> Qinv_ = Q.inverse();
            CMatrix<D,D> Qh_Qinvt_ = Q.adjoint()*Qinv_.transpose();
            CMatrix<D,N> Qinv_dx_ = Qinv_*dx_;

            //calculate groud state (see Eq. 3.13), not including (detQ)^{-1/2}
            CMatrix<D,N> P_Qinv_dx = P*Qinv_dx_;   //PQ^{-1}(x-q)
            CArray<1,N> pr1 = ( dx_.array() * P_Qinv_dx.array() ).colwise().sum();
            CArray<1,N> pr2 = ( p.transpose()*dx_ ).array();

            CArray<1,N> e = complex_t(0.0, 1.0)/(eps_*eps_) * (0.5*pr1 + pr2);
            CArray<1,N> phi_0=e.exp() / std::pow(math::pi<real_t>()*eps_*eps_, D/4.0);
            //******************************************************************************

            

            //recursively calculate all basis function values at grid node, not including (detQ)^{-1/2}
            HaWpBasisVector<N> complete_basis(shape_.size(), grid.cols());   //store the basis set \phi_k(x)  size of (number of basis)* (number of grid nodes)

            complete_basis.row(0)=phi_0;                                     //store the ground state 
            //CArray<1,N> pr1(1, N), pr2(1, N);   // pr1 store the current node value, pr2 store the previous node value (see Eq. 3,58)

            

            for(std::size_t j = 1; j < shape_.size(); j++){
            	std::array<int, D> next_index = shape_.get_item(j);   // get the index for k+e_d

            	//find valid precursor: find first non-zero entry
            	dim_t axis = D;
            	for (dim_t d = 0; d < D; d++) {
            		if (next_index[d] != 0) {
            			axis = d;
            			break;
            		}
            	}
            	assert(axis != D); //assert that multi-index contains some non-zero entries

            	// compute contribution of current node
            	std::array<int,D> curr_index = next_index;
            	curr_index[axis] -= 1; //get backward neighbour

            	int curr_ordinal=shape_.get_item(curr_index);    // given the current index, obtain the corresponding linear mapping 
            	pr1 = complete_basis.row(curr_ordinal) * Qinv_dx_.row(axis).array() * std::sqrt(2.0)/eps_ ;


            	// compute contribution of previous nodes 
            	std::array<std::array<int, D>,D> backward_neighbours=shape_.get_backward_neighbours(curr_index); //obtain the previous nodes 
            	int prev_ordinal;
            	pr2.setZero();
            	for (dim_t d = 0; d < D; d++) {
            		prev_ordinal=shape_.get_item(backward_neighbours[d]);

            		if(prev_ordinal !=-1){
            			pr2 += complete_basis.row(prev_ordinal) * Qh_Qinvt_(axis,d) * sqrt_[ curr_index[d] ];
            		} 
            	}

            	complete_basis.row(j) = (pr1 - pr2) / sqrt_[ 1+curr_index[axis] ];

            }

            return complete_basis;

        }

        /**
        * \brief Evaluates all basis functions \f$ \{\phi_k\} \f$ on real grid nodes \f$ x \in \gamma \f$.
        *
        * \param rgrid
        * Real grid nodes / quadrature points \f$ \gamma \f$.
        * Real matrix with shape (dimensionality, number of grid nodes).
        * \return Complex 2D-array with shape (basis shape size, number of grid nodes)
        *
        * \tparam N
        * Number of quadrature points.
        * Don't choose Eigen::Dynamic. It works, but performance is bad.
        */
        template<int N> HaWpBasisVector<N>
        evaluate_basis(RMatrix<D,N> const& rgrid) const
        {
            CMatrix<D,N> cgrid = rgrid.template cast <complex_t>();
            return evaluate_basis(cgrid);
        }

        /**
        * \brief Evaluates this wavepacket \f$ \Phi(x) \f$ at complex grid nodes \f$ x \in \gamma \f$.
        *
        * Notice that this function does not include the prefactor
        * \f$ \frac{1}{\sqrt{det(Q)}} \f$ nor the global phase
        * \f$ \exp{(\frac{iS}{\varepsilon^2})} \f$.
        *
        * \param grid
        * Complex grid nodes / quadrature points \f$ \gamma \f$.
        * Complex matrix with shape (dimensionality, number of grid nodes).
       	* \return Complex matrix with shape (1, number of grid nodes)
        */

        template<int N> CArray<1,N>
        evaluate(CMatrix<D,N> const& grid) const
        {
            if (shape_.size() != (std::size_t)coefficients_.size())
                 throw std::runtime_error("shape.size() != coefficients.size()");


            // use Kahan's algorithm to accumulate bases with O(1) numerical error instead of O(Sqrt(N))
             
             math::KahanSum< CArray<1,N> > psi( CArray<1,N>::Zero(1, grid.cols()) );

             HaWpBasisVector<N> complete_basis=evaluate_basis(grid);

             for(std::size_t ii=0; ii<shape_.size(); ii++)
             	psi += complete_basis.row(ii)*coefficients_[ii];

             return psi();
        }

        /**
        * \brief Evaluates this wavepacket \f$ \Phi(x) \f$ at real grid nodes \f$ x \in \gamma \f$.
        *
        * Notice that this function does not include the prefactor
        * \f$ \frac{1}{\sqrt{det(Q)}} \f$ nor the global phase
        * \f$ \exp{(\frac{iS}{\varepsilon^2})} \f$.
        *
        * \param rgrid
        * Real grid nodes / quadrature points \f$ \gamma \f$.
        * Real matrix with shape (dimensionality, number of grid nodes).
        * \return Complex matrix with shape (1, number of grid nodes)
        */
        template<int N> CArray<1,N>
        evaluate(RMatrix<D,N> const& rgrid) const
        {
            CMatrix<D,N> cgrid = rgrid.template cast <complex_t>();
            return evaluate(cgrid);
        }

        /**
        * \brief Compute the gradient y\Phi by scatter-type stencil application (Algorithm 9), can be used for the calculation of kinetic operator T, return the new coefficient cprime
        */

        CMatrix<D, Eigen::Dynamic> apply_gradient() const 
        {
            
            RMatrix<D,1> const& p = parameters_.p();
            CMatrix<D,D> const& P = parameters_.P();
            CMatrix<D,D> P_bar=P.conjugate();          //conjugate of the P

            auto extend_shape=shape_.extend();   //obtain the extended basis shape 

            dim_t size=shape_.size();                 //get the basis size of the current basis shape 
            dim_t size_extend=extend_shape.size();    //get the basis size of the extended basis shape 

            CMatrix<D, Eigen::Dynamic> cprime=CMatrix<D, Eigen::Dynamic>::Zero(D, size_extend);   //Storage space for the result 

            for(dim_t kk=0; kk<size; kk++){

                std::array<int, D> curr_index = shape_.get_item(kk);  //get the current index 

                cprime.col(extend_shape.get_item(curr_index))+=coefficients_(kk)*p;

                //compute the contribution from previous nodes and next nodes 
                int prev_ordinal, next_ordinal;
                std::array<std::array<int, D>,D> backward_neighbours=shape_.get_backward_neighbours(curr_index); //obtain the previous nodes 
                std::array<std::array<int, D>,D> forward_neighbours=shape_.get_forward_neighbours(curr_index);   //obtain the next nodes                                 
                for (dim_t d = 0; d < D; d++) {
                    prev_ordinal=extend_shape.get_item(backward_neighbours[d]);

                    if(prev_ordinal !=-1){
                        cprime.col(extend_shape.get_item(backward_neighbours[d]))+=eps_/std::sqrt(real_t(2))*std::sqrt(real_t(curr_index[d]))*coefficients_(kk)*P_bar.col(d);
                    } 

                    next_ordinal=extend_shape.get_item(forward_neighbours[d]);

                    if(next_ordinal !=-1){
                        cprime.col(extend_shape.get_item(forward_neighbours[d]))+=eps_/std::sqrt(real_t(2))*std::sqrt(real_t(curr_index[d]+1))*coefficients_(kk)*P.col(d);
                    }
                }
            }

            return cprime;

        }




        /**
        * \brief Grants writeable access to the semi-classical scaling parameter
        * \f$ \varepsilon \f$ of the wavepacket.
        */
        real_t & eps()
        {
            return eps_;
        }

        real_t eps() const
        {
            return eps_;
        }
            
        /**
        * \brief Grants writeable access to the Hagedorn parameter set
        * \f$ \Pi \f$ of the wavepacket.
        */
        HaWpParamSet<D> & parameters()
        {
            return parameters_;
        }

        HaWpParamSet<D> const& parameters() const
        {
            return parameters_;
        }
            
        /**
        * \brief Grants access to the basis shape
        * \f$ \mathfrak{K} \f$ of the wavepacket.
        */
        Shape & shape()
        {
        	return shape_;
        }

        Shape const& shape() const
        {
            return shape_;
        }
        
        /**
        * \brief Grants writeable access to the coefficients \f$ c \f$
        * of the wavepacket.
        */
        Coefficients & coefficients()
        {
            return coefficients_;
        }

        Coefficients const& coefficients() const
        {
            return coefficients_;
        }

        /**
        * \brief Computes the prefactor \f$ \frac{1}{\sqrt{det(Q)}} \f$.
        */
        complex_t prefactor() const
        {
                return real_t(1) / parameters_.sdQ();
        }

        /**
        * \brief Computes the global phase factor \f$ \exp{(\frac{i S}{\varepsilon^2})} \f$.
        */
        complex_t phasefactor() const
        {
            return std::exp(complex_t(0,1) * parameters_.S() / eps_ / eps_);
        }

    };


    /**
    * \brief Represents a vectorized Hagedorn wavepacket \f$ \Psi \f$
    * with \f$ N \f$ components \f$ \Phi_n \f$.
    * Here we combine both the homogenous and inhomogenous Hagedorn wavepacket together
    * The only difference between the homogenous and inhomogenous Hagedorn wavepacket is the way the wavepacket propagate 
    * The number of components is determined at runtime.
    *
    * \tparam D wavepacket dimensionality
    * \tparam packets:  N component wavepackets, tuple of N component wavepackets  
    */
    template<dim_t D, class... packets>
    class VectorHaWp
    {
    public:

        VectorHaWp() = default;  // default constructor 

        //Constructor 
        VectorHaWp(real_t eps, const std::tuple<packets...> & components)
        {
            eps_=eps;
            components_=components;
        }

        //copy constructor 
        VectorHaWp(const VectorHaWp& that){ 
            
            eps_=that.eps_;
            components_=that.components_;
        }
    
        VectorHaWp &operator=(const VectorHaWp& that){  //assignment operator
            
            eps_=that.eps_;
            components_=that.components_;
            return *this;
        }

        /**
        * \brief Grants access to the semi-classical scaling parameter
        * \f$ \varepsilon \f$ of the wavepacket.
        */
        real_t & eps()
        {
            return eps_;
        }

        /**
        * \brief Retrieves the semi-classical scaling parameter
        * \f$ \varepsilon \f$ of the wavepacket.
        */
        real_t eps() const
        {
            return eps_;
        }


        /**
        * \brief Grants writeable access to all components
        * \f$ \{\Phi_n\} \f$ of this wavepacket.
        */
        std::tuple<packets...>  & components()
        {
            return components_;
        }

        /**
        * \brief Grants read-only access to all components
        * \f$ \{\Phi_n\} \f$ of this wavepacket.
        */
        std::tuple<packets...> const& components() const
        {
            return components_;
        }

        /**
        * \brief Returns the number of components.
        */
        std::size_t n_components() const
        {
            return std::tuple_size<std::tuple<packets...>>::value;  
        } 


    private:
        real_t eps_;       //the semiclassical scaling parameter 
        std::tuple<packets...> components_;  //represent the Phi_n; n-th component of the vectorized Hagedorn wavepacket. 

    }; 

    /**
    * \brief Represents a vectorized Hagedorn wavepacket \f$ \Psi \f$
    * with \f$ 2 \f$ components \f$ \Phi_n \f$.
    * Here we combine both the homogenous and inhomogenous Hagedorn wavepacket together
    * The only difference between the homogenous and inhomogenous Hagedorn wavepacket is the way the wavepacket propagate 
    * The number of components is 2
    *
    * \tparam D wavepacket dimensionality
    * \tparam ScalarPacket1:  component 1
    * \tparam ScalarPacket2:  component 2
    */
    template<dim_t D, class ScalarPacket1, class ScalarPacket2>
    class VectorHaWp2
    {
    public:

        VectorHaWp2() = default;  // default constructor 

        //Constructor 
        VectorHaWp2(real_t eps, const std::tuple<ScalarPacket1, ScalarPacket2> & components)
        {
            eps_=eps;
            components_=components;
        }

        //copy constructor 
        VectorHaWp2(const VectorHaWp2& that){ 
            
            eps_=that.eps_;
            components_=that.components_;
        }
    
        VectorHaWp2 &operator=(const VectorHaWp2& that){  //assignment operator
            
            eps_=that.eps_;
            components_=that.components_;
            return *this;
        }

        /**
        * \brief Grants writeable access to the \f$ n \f$-th component
        * \f$ \Phi_n \f$.
        *
        * \param n The index \f$ n \f$ of the requested component (n=2)
        * \return Reference to the requested component.
        */

        decltype(auto) 
        component(std::size_t nn) 
        {
            if(nn==0){
                return std::get<0>(components_);
            }else{
                return std::get<1>(components_);
            }
        }

        /**
        * \brief Grants read-only access to the \f$ n \f$-th component
        * \f$ \Phi_n \f$.
        *
        * \param n The index \f$ n \f$ of the requested component. (n=2)
        * \return Reference to the requested component.
        */
        
        decltype(auto) 
        component(std::size_t nn) const
        {
            if(nn==0){
                return std::get<0>(components_);
            }else{
                return std::get<1>(components_);
            }
        }


        /**
        * \brief Returns the number of coefficients for all component of the Hagedorn wavepacket (2 components).
        */ 

        std::size_t size() const
        {

            std::size_t total_size=std::get<0>(components_).coefficients().size()+std::get<1>(components_).coefficients().size();
            return total_size;
        } 



        /**
        * \brief Returns the offset vector for the components of the hagedorn wavepackets (2 components)
        */        
        std::vector<dim_t>  offset() const
        {
            const dim_t n_comps = n_components();  //number of component of the hagedorn wavepacket 

            std::vector<dim_t> offsets(n_comps);
            offsets[0] = 0;
            offsets[1]=std::get<0>(components_).coefficients().size()+offsets[0];

            return offsets;
        }   

        /**
        * \brief Evaluate the value of all components at once.
        *
        * Evaluates \f$ \Psi(x) = \{\Phi_i(x)\} \f$,
        * where \f$ x \f$ is is a complex quadrature point.
        * Notice that this function does not include the prefactor
        * \f$ \frac{1}{\sqrt{det(Q)}} \f$ nor the global phase
        * \f$ \exp{(\frac{iS}{\varepsilon^2})} \f$ for each component of the vector Hagedorn wavepacket 
        *
        * \param grid
        * Complex quadrature points.
        * Complex matrix of shape (dimensionality, number of quadrature points)
        * \return
        * Complex matrix of shape (number of components, number of quadrature points)
        *
        * \tparam N
        * Number of quadrature points.
        * Don't use Eigen::Dynamic. It works, but performance is bad.
        */
        template<int N>
        CArray<Eigen::Dynamic,N> evaluate(CMatrix<D,N> const& grid) const
        {
            CArray<Eigen::Dynamic,N> result(n_components(),grid.cols());

            result.row(0)=std::get<0>(components_).evaluate(grid);
            result.row(1)=std::get<1>(components_).evaluate(grid);

            return result;
        }

        /**
        * \brief Evaluates the value of all components at once.
        *
        * Evaluates \f$ \Psi(x) = \{\Phi_i(x)\} \f$,
        * where \f$ x \f$ is is a real quadrature point.
        *
        * Notice that this function does not include the prefactor
        * \f$ \frac{1}{\sqrt{det(Q)}} \f$ nor the global phase
        * \f$ \exp{(\frac{iS}{\varepsilon^2})} \f$ for each component of the vector Hagedorn wavepacket 
        * \param rgrid
        * Real quadrature points.
        * Real matrix of shape (dimensionality, number of quadrature points)
        * \return
        * Complex matrix of shape (number of components, number of quadrature points)
        *
        * \tparam N
        * Number of quadrature points.
        * Don't use Eigen::Dynamic. It works, but performance is bad.
        */
        template<int N>
        CArray<Eigen::Dynamic,N> evaluate(RMatrix<D,N> const& rgrid) const
        {
            CMatrix<D,N> cgrid = rgrid.template cast<complex_t>();
            return evaluate(cgrid);
        }

        /**
        * \brief Grants access to the semi-classical scaling parameter
        * \f$ \varepsilon \f$ of the wavepacket.
        */
        real_t & eps()
        {
            return eps_;
        }

        /**
        * \brief Retrieves the semi-classical scaling parameter
        * \f$ \varepsilon \f$ of the wavepacket.
        */
        real_t eps() const
        {
            return eps_;
        }


        /**
        * \brief Grants writeable access to all components
        * \f$ \{\Phi_n\} \f$ of this wavepacket.
        */
        std::tuple<ScalarPacket1, ScalarPacket2>  & components()
        {
            return components_;
        }

        /**
        * \brief Grants read-only access to all components
        * \f$ \{\Phi_n\} \f$ of this wavepacket.
        */
        std::tuple<ScalarPacket1, ScalarPacket2> const& components() const
        {
            return components_;
        }

        /**
        * \brief Returns the number of components.
        */
        std::size_t n_components() const
        {
            return 2;  
        } 


    private:
        real_t eps_;       //the semiclassical scaling parameter 
        std::tuple<ScalarPacket1, ScalarPacket2> components_;  //represent the Phi_n; n-th component of the vectorized Hagedorn wavepacket. 

    };


}

