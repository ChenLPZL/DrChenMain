#pragma once

#include <cassert>
#include <cmath>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>  // this header defines a set of standard exceptions that both the library and programs can use to report common errors 
#include <string>
#include <tuple>
#include <vector>

#include "types.hpp"
#include "multi_index.hpp"


namespace shapes {

    /**
    * \brief Subclasses provide a description of a basis shape.
    *
    * A \f$ D \f$-dimensional basis shape \f$ \mathfrak{K} \f$ is a set
    * of _unordered_ \f$ D \f$ dimensional integer tuples (aka _node_).
    *
    * Subclasses provide an description of a basis shape
    * \f$ \mathfrak{K} \subset \mathbb{N}_0^D\f$.
    * It describes, which nodes \f$ \underline{k} \in \mathbb{N}_0^D \f$
    * are part of the shape.
    * Instances are passed to the ShapeEnumerator,
    * which converts the information to an ShapeEnum.
    *
    * Keep in mind, that basis shapes must fulfill the fundamental property
    * \f[
    * \underline{k} \in \mathfrak{K} \Rightarrow \forall
    * \underline{k}-\underline{e}^d \in \mathfrak{K} \;\forall d \in \{d \;|\;k_d \geq 1\}
    * \f]
    * where \f$ \underline{e}^d \f$ is the unit vector in direction \f$ d \f$.
    * That means, if an arbitrary node is part of the basis shape, then all nodes
    * in the backward cone are part of the shape too.
    *
    * \tparam D basis shape dimensionality
    */
    template<dim_t D>
    class AbstractShape
    {
    public:
        virtual ~AbstractShape() { }  //virtual destructor 

        //get the backward neighbours of multi_index index 
        std::array<TinyMultiIndex<D>,D>    
        get_backward_neighbours(const TinyMultiIndex<D> &index) const{

            std::array<TinyMultiIndex<D>,D> backward_neighbours;
            TinyMultiIndex<D> index_minus(index);

            for(std::size_t ii=0; ii<D; ii++){
                index_minus[ii]-=1;
                backward_neighbours.at(ii)=index_minus;
                index_minus[ii]+=1;
            }

            return backward_neighbours;

        }

        //get the forward neighbours of multi_index index 
        std::array<TinyMultiIndex<D>,D>    
        get_forward_neighbours(const TinyMultiIndex<D> &index) const{

            std::array<TinyMultiIndex<D>,D> forward_neighbours;
            TinyMultiIndex<D> index_plus(index);

            for(std::size_t ii=0; ii<D; ii++){
                index_plus[ii]+=1;
                forward_neighbours.at(ii)=index_plus;
                index_plus[ii]-=1;
            }

            return forward_neighbours;

        }

        /**
        * \brief Retrieves the length of the minimum bounding box in one direction.
        *
        * The minimum bounding box is given by
        * \f[
        * L_{\alpha}=\max_{k_{\alpha}}\left\{\underline{k} \in \mathfrak{K}\right\}
        * \f]
        *
        * \param axis The direction \f$ \alpha \f$.
        * \return Length of the bbox.
        */
        
        
        virtual int bbox(dim_t axis) const = 0;  //pure virtual member function 

        /**
        * \brief Evaluates the limit in direction axis given a base node.
        *
        * The surface function to direction \f$ \alpha \f$ is given by
        *
        * \f[
        * s_{\alpha}(\underline{n})=\max_{k_{\alpha}}
        * \left\{\underline{k} \in \mathfrak{K} \;|\;
        * k_d = n_d \; \forall d \neq \alpha
        * \right\}
        * \f]
        *
        * Notice that the \f$ \alpha \f$-th entry of \f$ \underline{n} \f$
        * does not influence return value.
        * It can be of any value since it is simply ignored.
        *
        * \param base_node The basis node \f$ \underline{n} \f$. It contains D indices.
        * \param axis The direction \f$ \alpha \f$.
        * \return the limit of direction axis
        */
        virtual int limit(int const* base_node, dim_t axis) const = 0;  // pure virtual function 

        /**
        * \brief Prints a pretty description of the shape.
        *
        * \param out The output stream.
        */
        virtual void print(std::ostream & out) const = 0; //pure virtual function

    };

    //overridding the operator << used for the output stream 
    template<dim_t D>
    std::ostream & operator<<(std::ostream & out, AbstractShape<D> const& shape)
    {
        shape.print(out);
        return out;
    }
    
    /**
    * \brief This class implements the hyperbolic cut shape.
    *
    * This class implements the hyperbolic cut basis shape, which is a special
    * type of a sparse basis shape.
    * The hyperbolic cut shape in \f$ D \f$ dimensions with _sparsity_ \f$S\f$
    * is defined as the set
    *
    * \f[
    * \mathfrak{K}(D,S) := \left\{(k_1,\dots,k_D) \in \mathbb{N}_0^D \mid
    *      \displaystyle\prod_{d=1}^{D} (1+k_d) \leq S \right\}
    * \f]
    *
    * \tparam D basis shape dimensionality
    */
    template<dim_t D>
    class HyperbolicCutShape : public AbstractShape<D>
    {
    private:
        int S_;   // the sparsity parameter 
        std::map<TinyMultiIndex<D>, int> lima_;        //linear map: MultiIndex--> int
        std::map<int,TinyMultiIndex<D>> lima_inv_;    //inverse linear map: int-->MultiIndex 
        std::size_t basis_size_;                               //number of basis function 
        std::vector<std::vector<TinyMultiIndex<D>>> slices_;    //rearrange the multi-index into slices where each slice has the same value for the sum of multi-index (i.e. the sth slice satisfy s=\sum_{i=1}^D{k}_i)

    public:


        HyperbolicCutShape() = default;  // default constructor 
    /**
    * \brief General constructor to set the sparsity parameter \f$ S \f$.
    *
    * \param S The sparsity parameter \f$ S \f$.
    */

        HyperbolicCutShape(int S) : S_(S) {
            auto pa=get_index_lex();
            lima_=std::get<0>(pa);
            lima_inv_=std::get<1>(pa);
            slices_=std::get<2>(pa);
            basis_size_=lima_.size();

        }
          
        HyperbolicCutShape(const HyperbolicCutShape& that){ //copy constructor 
            
            S_=that.S_;
            lima_=that.lima_;
            lima_inv_=that.lima_inv_;
            basis_size_=that.basis_size_;
            slices_=that.slices_;
        }

        HyperbolicCutShape(HyperbolicCutShape&& that)    //move copy constructor 
            : S_(that.S_)
            , lima_(std::move(that.lima_))
            , lima_inv_(std::move(that.lima_inv_))
            , basis_size_(that.basis_size_)
            , slices_(std::move(that.slices_))
        {}

        
        HyperbolicCutShape &operator=(const HyperbolicCutShape& that){  //assignment operator
            
            S_ = that.S_;
            lima_=that.lima_;
            lima_inv_ = that.lima_inv_;
            basis_size_=that.basis_size_;
            slices_=that.slices_;
            return *this;
        } 

        HyperbolicCutShape &operator=(HyperbolicCutShape&& that)   //move assignment operator 
        {
            
            S_ = that.S_;
            lima_=std::move(that.lima_);
            lima_inv_ = std::move(that.lima_inv_);
            basis_size_=that.basis_size_;
            slices_=std::move(that.slices_);
            return *this;
        }

        //Given the MultiIndex, get the corresponding linear mapping.
        int& get_item(const TinyMultiIndex<D> &index){

            int * value_p=new int;
            *value_p=-1;

            if(contains(index)){
               return lima_.at(index);  
            }else{
                return *value_p;  //not found, return -1
            }
            
        }

        const int& get_item(const TinyMultiIndex<D> &index) const{

            int * value_p=new int;
            *value_p=-1;

            if(contains(index)){
               return lima_.at(index);  
            }else{
                return *value_p;  //not found, return -1
            }

        }

        //Given mapped int value, get the corresponding MultiIndex 
        TinyMultiIndex<D>& get_item(const int &kk){


            assert(kk>=0 && kk<lima_inv_.size());

            return lima_inv_.at(kk);
            
        }


        //Given mapped int value, get the corresponding MultiIndex 
        const TinyMultiIndex<D>& get_item(const int &kk) const{

            assert(kk>=0 && kk<lima_inv_.size());

            return lima_inv_.at(kk);

        }

        //check if a given multi-index is part of the basis set
        bool contains(const TinyMultiIndex<D> &index) const{

            if(lima_.count(index)>0)
                return true;
            else
                return false;
        }


        HyperbolicCutShape extend() const {

            int extended_sparsity;
            if(D>1)
                extended_sparsity=std::pow(2, D-1)*S_;
            else
                extended_sparsity=S_+1;

            return std::move(HyperbolicCutShape(extended_sparsity));
        }

        std::tuple<std::map<TinyMultiIndex<D>,int>, std::map<int,TinyMultiIndex<D>>, std::vector<std::vector<TinyMultiIndex<D>>>>
        get_index_lex(){

            // enumerate shape and store all multi-indices
            
                TinyMultiIndex<D> index{}; //zero initialize
                std::vector<TinyMultiIndex<D>> mindices;   //vector storing multi-index 

                while (true) {
                    // iterate over last axis
                    for (dim_t i = 0; i <= limit(index.data(),D-1); i++) {
                        index[D-1] = i;

                        mindices.push_back(index);
                    }
                    index[D-1] = 0;

                    // iterate over other axes
                    if (D > 1) {
                        dim_t j = D-2;
                        while ((int)index[j] == limit(index.data(),j)) {
                            index[j] = 0;
                            if (j == 0)
                                goto enumeration_complete;
                            else
                                j = j-1;
                        }
                        index[j] += 1;
                    }
                    else break;
                }
                enumeration_complete:
                (void)0;
            

            std::size_t max_size=mindices.size();
            std::map<TinyMultiIndex<D>,int>  lima;
            std::map<int,TinyMultiIndex<D>>  lima_inv;


            std::vector<std::vector<TinyMultiIndex<D>>> slices;

            std::size_t sum=0;
            for(auto& indice:mindices){
                if(sum<=std::accumulate(indice.begin(), indice.end(), 0))
                    sum=std::accumulate(indice.begin(), indice.end(), 0);
            }
                        
            slices.resize(sum+1);


            for(int ii=0; ii<max_size; ii++){
                lima[mindices[ii]]=ii;
                lima_inv[ii]=mindices[ii];
                slices[std::accumulate(mindices[ii].begin(), mindices[ii].end(), 0)].push_back(mindices[ii]);
            }

            return std::make_tuple(lima, lima_inv, slices);

        }


        std::map<TinyMultiIndex<D>, int>& get_lima() {

            return lima_;
        }

        const std::map<TinyMultiIndex<D>, int>& get_lima() const{

            return lima_;
        }

        std::map<int,TinyMultiIndex<D>>& get_lima_inv() {

            return lima_inv_;
        }

        const std::map<int,TinyMultiIndex<D>>& get_lima_inv() const{

            return lima_inv_;
        }

        std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() {

            return slices_;
        }

        const std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() const{

            return slices_;
        }

        std::size_t size() const {
            return basis_size_;
        }

        int sparsity() const {

        	return S_;
        }

        virtual int bbox(dim_t axis) const override
        {
            (void)(axis); // unused
            return S_ - 1;
        }

        virtual int limit(int const* base_node, dim_t axis) const override
        {
            double s = S_;

            for (dim_t i = 0; i < D; i++) {
                if (i != axis) {
                    s /= 1 + base_node[i];
                }
            }

            return (int)s - 1;
        }

        virtual void print(std::ostream & out) const override
        {
            out << "HyperbolicCutShape{sparsity: " << S_ << "}";
        }


    };

    /**
    * \brief This class implements the limited hyperbolic cut shape.
    *
    * This class implements the limited hyperbolic cut basis shape which is a special
    * type of a sparse basis shape.
    * The limited hyperbolic cut shape in \f$ D \f$ dimensions with _sparsity_ \f$S\f$ and
    * _limits_ \f$ \boldsymbol{K} = (K_1,\ldots,K_D) \f$ is defined as the set
    *
    * \f[
    * \mathfrak{K}(D,S,\boldsymbol{K}) := \left\{(k_1,\dots,k_D) \in \mathbb{N}_0^D \mid
    *      0 \leq k_d < K_d \; \land
    *      \displaystyle\prod_{d=1}^{D} (1+k_d) \leq S \right\}
    * \f]
    *
    * It is an intersection of the hyperbolic cut shape with a hypercubic shape.
    *
    * \tparam D basis shape dimensionality
    */
    template<dim_t D>
    class LimitedHyperbolicCutShape : public AbstractShape<D>
    {
    private:
        int S_;                     // the sparsity 
        std::array<int,D> limits_;  //  the limits for each dimension 

        std::map<TinyMultiIndex<D>, int> lima_;        //linear map: MultiIndex--> int
        std::map<int,TinyMultiIndex<D>> lima_inv_;    //inverse linear map: int-->MultiIndex 
        std::size_t basis_size_;     //number of basis function 
        std::vector<std::vector<TinyMultiIndex<D>>> slices_;    //rearrange the multi-index into slices where each slice has the same value for the sum of multi-index (i.e. the sth slice satisfy s=\sum_{i=1}^D{k}_i)

    public:


    	LimitedHyperbolicCutShape()=default;   // default constructor 
        /**
        * \brief General constructor to define sparsity parameter and limits.
        *
        * \param S The sparsity parameter \f$ S \f$.
        * \param limits Tuple of all limits \f$ \boldsymbol{K} \f$.
        */
        LimitedHyperbolicCutShape(int S, const std::array<int,D> &limits)
            : S_(S)
            , limits_(limits){

            	auto pa=get_index_lex();
            	lima_=std::get<0>(pa);
            	lima_inv_=std::get<1>(pa);
            	slices_=std::get<2>(pa);
            	basis_size_=lima_.size();
            }

        /**
        * \brief Specialized constructor to set all limits \f$ K_d \f$ to the same value \f$ K^\star \f$.
        *
        * \param S The sparsity parameter \f$ S \f$.
        * \param size The limit \f$ K^\star \f$.
        */
        LimitedHyperbolicCutShape(int S, int size)
            : S_(S)
        {
            for (std::size_t d = 0; d < D; d++)
                limits_[d] = size;

            auto pa=get_index_lex();
            lima_=std::get<0>(pa);
            lima_inv_=std::get<1>(pa);
            slices_=std::get<2>(pa);
            basis_size_=lima_.size();

        }

        /**
        * \brief General constructor to define sparsity parameter and limits.
        *
        * \param S The sparsity parameter \f$ S \f$.
        * \param list List of all limits \f$ \boldsymbol{K} \f$.
        */
        LimitedHyperbolicCutShape(int S, std::initializer_list<int> list)
            : S_(S)
        {
            int deflt = 0;
            std::size_t i = 0;
            for (int e : list) {
                limits_[i++] = deflt = e;
            }
            //fill remaining elements with last value of initializer list
            while (i < D) {
                limits_[i++] = deflt;
            }

            auto pa=get_index_lex();
            lima_=std::get<0>(pa);
            lima_inv_=std::get<1>(pa);
            slices_=std::get<2>(pa);
            basis_size_=lima_.size();

        }

        // copy constructor 
        LimitedHyperbolicCutShape(const LimitedHyperbolicCutShape& that){

        	S_=that.S_;
        	limits_=that.limits_;
        	lima_=that.lima_;
        	lima_inv_=that.lima_inv_;
        	basis_size_=that.basis_size_;
        	slices_=that.slices_;
        }

        //move copy constructor 
        LimitedHyperbolicCutShape(LimitedHyperbolicCutShape&& that){

        	S_=that.S_;
        	limits_=that.limits_;
        	lima_=std::move(that.lima_);
        	lima_inv_=std::move(that.lima_inv_);
        	basis_size_=that.basis_size_;
        	slices_=std::move(that.slices_);
        }

        //assignment operator 
        LimitedHyperbolicCutShape &operator=(const LimitedHyperbolicCutShape& that){

        	S_=that.S_;
        	limits_=that.limits_;
        	lima_=that.lima_;
        	lima_inv_=that.lima_inv_;
        	basis_size_=that.basis_size_;
        	slices_=that.slices_;
        	return *this;

        }

        //move assignment operator 
        LimitedHyperbolicCutShape &operator=(LimitedHyperbolicCutShape&& that){

        	S_=that.S_;
        	limits_=that.limits_;
        	lima_=std::move(that.lima_);
        	lima_inv_=std::move(that.lima_inv_);
        	basis_size_=that.basis_size_;
        	slices_=std::move(that.slices_);
        	return *this;
        }

        //Given the MultiIndex, get the corresponding linear mapping.
        int& get_item(const TinyMultiIndex<D> &index){

        	int * value_p=new int;
        	*value_p=-1;

        	if(contains(index)){
        		return lima_.at(index);
        	}else{
        		return *value_p;   //not found, return -1
        	}
        }

        const int& get_item(const TinyMultiIndex<D> &index) const{

        	int * value_p=new int;
        	*value_p=-1;

        	if(contains(index)){
        		return lima_.at(index);
        	}else{
        		return *value_p;   //not found, return -1
        	}
        }

        //Given mapped int value, get the corresponding MultiIndex 
        TinyMultiIndex<D>& get_item(const int &kk){

        	assert(kk>=0 && kk<lima_inv_.size());
        	return lima_inv_.at(kk);
        }

        const TinyMultiIndex<D>& get_item(const int &kk) const{

        	assert(kk>=0 && kk<lima_inv_.size());

        	return lima_inv_.at(kk);
        }

        //check if a given multi-index is part of the basis set
        bool contains(const TinyMultiIndex<D> &index) const{

        	if(lima_.count(index)>0)
        		return true;
        	else
        		return false;
        }

        LimitedHyperbolicCutShape extend() const {
            
            int extended_sparsity;
            std::array<int,D> extended_limits(limits_);

            if(D>1){
                extended_sparsity=std::pow(2, D-1)*S_;
            }
            else{
                extended_sparsity=S_+1;
            }

            for(std::size_t ii=0; ii<D; ii++)
                extended_limits[ii]=limits_[ii]+1;

            return std::move(LimitedHyperbolicCutShape(extended_sparsity, extended_limits));

        }


        std::tuple<std::map<TinyMultiIndex<D>,int>, std::map<int,TinyMultiIndex<D>>, std::vector<std::vector<TinyMultiIndex<D>>>>
        get_index_lex(){

            // enumerate shape and store all multi-indices
            
                TinyMultiIndex<D> index{}; //zero initialize
                std::vector<TinyMultiIndex<D>> mindices;   //vector storing multi-index 

                while (true) {
                    // iterate over last axis
                    for (dim_t i = 0; i <= limit(index.data(),D-1); i++) {
                        index[D-1] = i;

                        mindices.push_back(index);
                    }
                    index[D-1] = 0;

                    // iterate over other axes
                    if (D > 1) {
                        dim_t j = D-2;
                        while ((int)index[j] == limit(index.data(),j)) {
                            index[j] = 0;
                            if (j == 0)
                                goto enumeration_complete;
                            else
                                j = j-1;
                        }
                        index[j] += 1;
                    }
                    else break;
                }
                enumeration_complete:
                (void)0;
            

            std::size_t max_size=mindices.size();
            std::map<TinyMultiIndex<D>,int>  lima;
            std::map<int,TinyMultiIndex<D>>  lima_inv;


            std::vector<std::vector<TinyMultiIndex<D>>> slices;

            std::size_t sum=0;
            for(auto& indice:mindices){
                if(sum<=std::accumulate(indice.begin(), indice.end(), 0))
                    sum=std::accumulate(indice.begin(), indice.end(), 0);
            }
                        
            slices.resize(sum+1);


            for(int ii=0; ii<max_size; ii++){
                lima[mindices[ii]]=ii;
                lima_inv[ii]=mindices[ii];
                slices[std::accumulate(mindices[ii].begin(), mindices[ii].end(), 0)].push_back(mindices[ii]);
            }

            return std::make_tuple(lima, lima_inv, slices);
        }

        std::map<TinyMultiIndex<D>, int>& get_lima() {

            return lima_;
        }

        const std::map<TinyMultiIndex<D>, int>& get_lima() const{

            return lima_;
        }

        std::map<int,TinyMultiIndex<D>>& get_lima_inv() {

            return lima_inv_;
        }

        const std::map<int,TinyMultiIndex<D>>& get_lima_inv() const{

            return lima_inv_;
        }

        std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() {

            return slices_;
        }

        const std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() const{

            return slices_;
        }

        std::size_t size() const {
            return basis_size_;
        }

        int sparsity() const {

            return S_;
        }

        std::array<int,D>& get_limits() {

        	return limits_;
        }

        const std::array<int,D>& get_limits() const{

        	return limits_;
        }



        virtual int bbox(dim_t axis) const override
        {
            return std::min( limits_[axis]-1, S_ - 1);
        }

        virtual int limit(int const* base_node, dim_t axis) const override
        {
           	double s = S_;

           	for (dim_t i = 0; i < D; i++) {
                if (i != axis) {
                    if (base_node[i] >= limits_[i])
                        return -1;
                    else
                        s /= 1 + base_node[i];
                }
            }

            return std::min((int)limits_[axis]-1, (int)s - 1);
        }

       	virtual void print(std::ostream & out) const override
        {
            out << "LimitedHyperbolicCutShape{ sparsity: " << S_ << ", limits (exclusive): [";
            for (dim_t i = 0; i < D-1; i++) {
                 out << limits_[i] << ",";
            }
            out << limits_[D-1] << "]";
            out << "}";
        }
    };


    /**
    * \ingroup BasicShape
    *
    * \brief This class implements the hypercubic basis shape.
    *
    * A \f$ D \f$-dimensional hypercubic shape with limits \f$ \mathbf{K}=\{K_1,\dots,K_D\} \f$ is defined as the set
    *
    * \f[
    * \mathfrak{K}(D,\mathbf{K}) := \left\{ (k_1, \dots, k_D) \in \mathbb{N}_0^D | k_d < K_d \forall d \right\}
    * \f]
    *
    * \tparam D basis shape dimensionality
    */
    template<dim_t D>
    class HyperCubicShape : public AbstractShape<D>
    {
    private:
        std::array<int,D> limits_;

        std::map<TinyMultiIndex<D>, int> lima_;        //linear map: MultiIndex--> int
        std::map<int,TinyMultiIndex<D>> lima_inv_;    //inverse linear map: int-->MultiIndex 
        std::size_t basis_size_;     //number of basis function 
        std::vector<std::vector<TinyMultiIndex<D>>> slices_;    //rearrange the multi-index into slices where each slice has the same value for the sum of multi-index (i.e. the sth slice satisfy s=\sum_{i=1}^D{k}_i)

    public:

    	HyperCubicShape()=default;   //default constructor 
        /**
        * \param[in] limits Array of all limits \f$ \{K_d\} \f$.
        */
        HyperCubicShape(const std::array<int,D> &limits)
            : limits_(limits)
        { 
        	auto pa=get_index_lex();
        	lima_=std::get<0>(pa);
        	lima_inv_=std::get<1>(pa);
        	slices_=std::get<2>(pa);
        	basis_size_=lima_.size();
        }

        /**
        * \brief Set limits to \f$ K_d := K \; \forall d \f$.
        *
        * \param limit Limit \f$ K \f$.
        */
        
        HyperCubicShape(int limit)
        {
            for (std::size_t d = 0; d < D; d++)
                        limits_[d] = limit;

        	auto pa=get_index_lex();
        	lima_=std::get<0>(pa);
        	lima_inv_=std::get<1>(pa);
        	slices_=std::get<2>(pa);
        	basis_size_=lima_.size();

        }

        /**
        * \param list List of all limits \f$ \{K_d\} \f$.
        */
        HyperCubicShape(std::initializer_list<int> list)
        {
            int deflt = 0;
            std::size_t i = 0;
            for (int e : list) {
                limits_[i++] = deflt = e;
            }
            //fill remaining elements with last value of initializer list
            while (i < D) {
                limits_[i++] = deflt;
            }

        	auto pa=get_index_lex();
        	lima_=std::get<0>(pa);
        	lima_inv_=std::get<1>(pa);
        	slices_=std::get<2>(pa);
        	basis_size_=lima_.size();
        }

        //copy constructor 
        HyperCubicShape(const HyperCubicShape &that)
            : limits_(that.limits_)
        { 
            lima_=that.lima_;
            lima_inv_=that.lima_inv_;
            basis_size_=that.basis_size_;
            slices_=that.slices_;
        }

        //move copy constructor 
        HyperCubicShape(HyperCubicShape&& that){

        	limits_=that.limits_;
            lima_=std::move(that.lima_);
            lima_inv_=std::move(that.lima_inv_);
            basis_size_=that.basis_size_;
            slices_=std::move(that.slices_);
        }

        //assignment operator 
        HyperCubicShape &operator=(const HyperCubicShape &that)
        {
            limits_ = that.limits_;
            lima_=that.lima_;
            lima_inv_=that.lima_inv_;
            basis_size_=that.basis_size_;
            slices_=that.slices_;
            return *this;
        }

        //copy assignment operator 
        HyperCubicShape &operator=(HyperCubicShape&& that){

        	limits_=that.limits_;
            lima_=std::move(that.lima_);
            lima_inv_=std::move(that.lima_inv_);
            basis_size_=that.basis_size_;
            slices_=std::move(that.slices_);
            return *this;

        }


        //Given the MultiIndex, get the corresponding linear mapping.
        int& get_item(const TinyMultiIndex<D> &index){

            int * value_p=new int;
            *value_p=-1;

            if(contains(index)){
                return lima_.at(index);
            }else{
                return *value_p;   //not found, return -1
            }
        }

        const int& get_item(const TinyMultiIndex<D> &index) const{

            int * value_p=new int;
            *value_p=-1;

            if(contains(index)){
                return lima_.at(index);
            }else{
                return *value_p;   //not found, return -1
            }
        }

        //Given mapped int value, get the corresponding MultiIndex 
        TinyMultiIndex<D>& get_item(const int &kk){

            assert(kk>=0 && kk<lima_inv_.size());
            return lima_inv_.at(kk);
        }

        const TinyMultiIndex<D>& get_item(const int &kk) const{

            assert(kk>=0 && kk<lima_inv_.size());

            return lima_inv_.at(kk);
        }

        //check if a given multi-index is part of the basis set
        bool contains(const TinyMultiIndex<D> &index) const{

            if(lima_.count(index)>0)
                return true;
            else
                return false;
        }


       HyperCubicShape extend() const {

       		std::array<int,D> extended_limits(limits_);
            for(std::size_t ii=0; ii<D; ii++)
                extended_limits[ii]=limits_[ii]+1;

            return std::move(HyperCubicShape(extended_limits));
       }

        std::tuple<std::map<TinyMultiIndex<D>,int>, std::map<int,TinyMultiIndex<D>>, std::vector<std::vector<TinyMultiIndex<D>>>>
        get_index_lex(){

            // enumerate shape and store all multi-indices
            
                TinyMultiIndex<D> index{}; //zero initialize
                std::vector<TinyMultiIndex<D>> mindices;   //vector storing multi-index 

                while (true) {
                    // iterate over last axis
                    for (dim_t i = 0; i <= limit(index.data(),D-1); i++) {
                        index[D-1] = i;

                        mindices.push_back(index);
                    }
                    index[D-1] = 0;

                    // iterate over other axes
                    if (D > 1) {
                        dim_t j = D-2;
                        while ((int)index[j] == limit(index.data(),j)) {
                            index[j] = 0;
                            if (j == 0)
                                goto enumeration_complete;
                            else
                                j = j-1;
                        }
                        index[j] += 1;
                    }
                    else break;
                }
                enumeration_complete:
                (void)0;
            

            std::size_t max_size=mindices.size();
            std::map<TinyMultiIndex<D>,int>  lima;
            std::map<int,TinyMultiIndex<D>>  lima_inv;


            std::vector<std::vector<TinyMultiIndex<D>>> slices;

            std::size_t sum=0;
            for(auto& indice:mindices){
                if(sum<=std::accumulate(indice.begin(), indice.end(), 0))
                    sum=std::accumulate(indice.begin(), indice.end(), 0);
            }
                        
            slices.resize(sum+1);


            for(int ii=0; ii<max_size; ii++){
                lima[mindices[ii]]=ii;
                lima_inv[ii]=mindices[ii];
                slices[std::accumulate(mindices[ii].begin(), mindices[ii].end(), 0)].push_back(mindices[ii]);
            }

            return std::make_tuple(lima, lima_inv, slices);
        }

         std::map<TinyMultiIndex<D>, int>& get_lima() {

            return lima_;
        }

        const std::map<TinyMultiIndex<D>, int>& get_lima() const{

            return lima_;
        }

        std::map<int,TinyMultiIndex<D>>& get_lima_inv() {

            return lima_inv_;
        }

        const std::map<int,TinyMultiIndex<D>>& get_lima_inv() const{

            return lima_inv_;
        }

        std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() {

            return slices_;
        }

        const std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() const{

            return slices_;
        }

        std::size_t size() const {
            return basis_size_;
        }

        std::array<int,D>& get_limits() {

            return limits_;
        }

        const std::array<int,D>& get_limits() const{

            return limits_;
        }


        virtual int limit(int const* base_node, dim_t axis) const override
        {
            { (void)(base_node); } //disable unused-parameter warning

            for (dim_t d = 0; d < D; d++) {
                if (d != axis && base_node[d] >= limits_[d])
                    return -1;
            }
            return limits_[axis]-1;
        }

        virtual int bbox(dim_t axis) const override
        {
            return limits_[axis]-1;
        }

        virtual void print(std::ostream & out) const override
        {
            out << "HyperCubicShape{ limits (exclusive): [";
            for (dim_t i = 0; i < D-1; i++) {
                out << limits_[i] << ",";
            }
            out << limits_[D-1] << "]";
            out << "}";
        }
    };

    /**
    * \ingroup BasicShape
    *
    * \brief This class implements the simplex basis shape .
    *
    * A \f$ D \f$-dimensional simplex shape with maximal 1-norm K is defined as the set
    *
    * \f[
    * \mathfrak{K}(D,\mathbf{K}) := \left\{ (k_1, \dots, k_D) \in \mathbb{N}_0^D | \sum_{d=1}^Dk_d<=K \right\}
    * \f]
    *
    * \tparam D basis shape dimensionality
    */

    template<dim_t D>
    class SimplexShape : public AbstractShape<D>
    {
    private:
    	int K_;       // maximal 1-norm K

        std::map<TinyMultiIndex<D>, int> lima_;        //linear map: MultiIndex--> int
        std::map<int,TinyMultiIndex<D>> lima_inv_;    //inverse linear map: int-->MultiIndex 
        std::size_t basis_size_;     //number of basis function 
        std::vector<std::vector<TinyMultiIndex<D>>> slices_;    //rearrange the multi-index into slices where each slice has the same value for the sum of multi-index (i.e. the sth slice satisfy s=\sum_{i=1}^D{k}_i)

    public:

    	SimplexShape()=default;   //default constructor 

    	SimplexShape(int K)  // constructor 
    	{
    		K_=K;

            auto pa=get_index_lex();
            lima_=std::get<0>(pa);
            lima_inv_=std::get<1>(pa);
            slices_=std::get<2>(pa);
            basis_size_=lima_.size();
    	}

    	//copy constructor 
    	SimplexShape(const SimplexShape& that){

    		K_=that.K_;
            lima_=that.lima_;
            lima_inv_=that.lima_inv_;
            basis_size_=that.basis_size_;
            slices_=that.slices_;   		

    	}

    	//move copy constructor 
    	SimplexShape(SimplexShape&& that){

    		K_=that.K_;
            lima_=std::move(that.lima_);
            lima_inv_=std::move(that.lima_inv_);
            basis_size_=that.basis_size_;
            slices_=std::move(that.slices_); 

    	}

    	//assignment operator 
    	SimplexShape &operator=(const SimplexShape& that){

    		K_=that.K_;
            lima_=that.lima_;
            lima_inv_=that.lima_inv_;
            basis_size_=that.basis_size_;
            slices_=that.slices_; 
            return *this;

    	}

    	SimplexShape &operator=(SimplexShape&& that){

    		K_=that.K_;
            lima_=std::move(that.lima_);
            lima_inv_=std::move(that.lima_inv_);
            basis_size_=that.basis_size_;
            slices_=std::move(that.slices_); 
            return *this;   		
    	}

        //Given the MultiIndex, get the corresponding linear mapping.
        int& get_item(const TinyMultiIndex<D> &index){

            int * value_p=new int;
            *value_p=-1;

            if(contains(index)){
                return lima_.at(index);
            }else{
                return *value_p;   //not found, return -1
            }
        }

        const int& get_item(const TinyMultiIndex<D> &index) const{

            int * value_p=new int;
            *value_p=-1;

            if(contains(index)){
                return lima_.at(index);
            }else{
                return *value_p;   //not found, return -1
            }
        }

        //Given mapped int value, get the corresponding MultiIndex 
        TinyMultiIndex<D>& get_item(const int &kk){

            assert(kk>=0 && kk<lima_inv_.size());
            return lima_inv_.at(kk);
        }

        const TinyMultiIndex<D>& get_item(const int &kk) const{

            assert(kk>=0 && kk<lima_inv_.size());

            return lima_inv_.at(kk);
        }

        //check if a given multi-index is part of the basis set
        bool contains(const TinyMultiIndex<D> &index) const{

            if(lima_.count(index)>0)
                return true;
            else
                return false;
        }

        SimplexShape extend() const {

        	return std::move(SimplexShape(K_+1));
        }

        std::tuple<std::map<TinyMultiIndex<D>,int>, std::map<int,TinyMultiIndex<D>>, std::vector<std::vector<TinyMultiIndex<D>>>>
        get_index_lex(){

            // enumerate shape and store all multi-indices
            
                TinyMultiIndex<D> index{}; //zero initialize
                std::vector<TinyMultiIndex<D>> mindices;   //vector storing multi-index 

                while (true) {
                    // iterate over last axis
                    for (dim_t i = 0; i <= limit(index.data(),D-1); i++) {
                        index[D-1] = i;

                        mindices.push_back(index);
                    }
                    index[D-1] = 0;

                    // iterate over other axes
                    if (D > 1) {
                        dim_t j = D-2;
                        while ((int)index[j] == limit(index.data(),j)) {
                            index[j] = 0;
                            if (j == 0)
                                goto enumeration_complete;
                            else
                                j = j-1;
                        }
                        index[j] += 1;
                    }
                    else break;
                }
                enumeration_complete:
                (void)0;
            

            std::size_t max_size=mindices.size();
            std::map<TinyMultiIndex<D>,int>  lima;
            std::map<int,TinyMultiIndex<D>>  lima_inv;


            std::vector<std::vector<TinyMultiIndex<D>>> slices;

            std::size_t sum=0;
            for(auto& indice:mindices){
                if(sum<=std::accumulate(indice.begin(), indice.end(), 0))
                    sum=std::accumulate(indice.begin(), indice.end(), 0);
            }
                        
            slices.resize(sum+1);


            for(int ii=0; ii<max_size; ii++){
                lima[mindices[ii]]=ii;
                lima_inv[ii]=mindices[ii];
                slices[std::accumulate(mindices[ii].begin(), mindices[ii].end(), 0)].push_back(mindices[ii]);
            }

            return std::make_tuple(lima, lima_inv, slices);
        }

        std::map<TinyMultiIndex<D>, int>& get_lima() {

            return lima_;
        }

        const std::map<TinyMultiIndex<D>, int>& get_lima() const{

            return lima_;
        }

        std::map<int,TinyMultiIndex<D>>& get_lima_inv() {

            return lima_inv_;
        }

        const std::map<int,TinyMultiIndex<D>>& get_lima_inv() const{

            return lima_inv_;
        }

        std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() {

            return slices_;
        }

        const std::vector<std::vector<TinyMultiIndex<D>>>& get_slices() const{

            return slices_;
        }

        std::size_t size() const {
            return basis_size_;
        }

        int max_norm() const {
        	return K_;
        }

        virtual int bbox(dim_t axis) const override
        {
            return K_;
        }

        virtual int limit(int const* base_node, dim_t axis) const override
        {

            int K= K_;

            for (dim_t i = 0; i < D; i++) {
                if (i != axis) {
                    if (base_node[i] > K_)
                        return -1;
                    else
                        K=K-base_node[i];
                }
            }

            return std::max(K,0);
        }

        virtual void print(std::ostream & out) const override
        {

        	out << "SimplexShape{ max_norm: "<<K_<<"}";
        }

    };
    
}

