#pragma once

#include <complex>
#include <stdexcept>  // this header defines a set of standard exceptions that both the library and programs can use to report common errors 

namespace math{

    template<class T>
    constexpr T pi()  // the constexpr specifier declares that it is possible to evaluate the value of the function of variable at compile time 
    {
        return static_cast<T>(3.141592653589793238462643383279);  //converts between types using a combination of implicit and user-defined conversions 
    }

    /**
    * \brief This class deals with the issue, that the square root of complex numbers is not unique.
    *
    * The equation \f$ z^2 = r \exp{(i\phi)} \f$ has two solutions, namely
    * \f$ z_1=\sqrt{r} \exp{\left(i\frac{\phi}{2}\right)} \f$ and
    * \f$ z_2=\sqrt{r} \exp{\left(i(\frac{\phi}{2}+\pi)\right)} \f$.
    *
    * This class chooses the solution, that is nearest to the solution of the previous computation ( = reference solution).
    * Then this class overrides the stored reference solution with the current solution.
    *
    * The distance between the two complex numbers is determined by the angle-distance.
    *
    * \tparam T Type of both the real and imaginary components of the complex number.
    */
        
    template<class T>
    class ContinuousSqrt
    {
    private:
        /**
        * stored reference solution
        */
        std::complex<T> sqrt_;  

        /**
        * angle of det(Q)
        */
        T state_;

        /**
        * false if a reference solution is stored
        */
        bool empty_;

    public:
        /**
        * \brief Delayes initialization of the stored reference solution.
        *
        * The next call to operator()() yields the principal square root.
        */
        ContinuousSqrt()
            : sqrt_()
            , state_()
            , empty_(true)
        { }

        /**
        * \brief Initializes the stored reference solution to a chosen value.
        *
        * \param sqrt The initial reference solution.
        */
        
        ContinuousSqrt(const std::complex<T> &detQ)
        { 
            sqrt_=operator()(detQ);
            state_=std::arg(detQ);
            empty_=false;
        }


        ContinuousSqrt(const ContinuousSqrt &that)
            : sqrt_(that.sqrt_)
            , state_(that.state_)
            , empty_(that.empty_)
        { }

        ContinuousSqrt &operator=(const ContinuousSqrt &that)
        {
            sqrt_ = that.sqrt_;
            state_ = that.state_;
            empty_ = that.empty_;
            return *this;
        }

        /**
        * Chooses the square root angle (argument) that continuates the reference angle the best.
        * Throws an exception if the deviation above an accepted value (by default > pi/4).
        * \param[in] ref The angle of the reference square root. domain = \f$ [-\pi;\pi] \f$
        * \param[in] arg The angle of the computed square root. domain = \f$ [-\pi;\pi] \f$
        * \return The angle of the continuating square root. domain = \f$ [-\pi;\pi] \f$
        */
        
        static T continuate(T ref, T arg)
        {
            const T PI = pi<T>();
            const T jump = 2.0*PI;

            T offset=round((arg-ref)/(1.0*jump));

            return arg-jump*offset;
        }

        /**
        * \brief Solves the quadratic equation \f$ z^2 = c \f$.
        *
        * Chooses the solution \f$ \hat{z} \f$ that best continuates the prior
        * result \f$ z_0 \f$ and updates the reference solution (\f$ z_0 \gets \hat{z} \f$).
        *
        * \param input The right-hand-side \f$ c \f$.
        * \return The best solution \f$ \hat{z} \f$.
        */
        
        std::complex<T> operator()(std::complex<T> input)
        {
            if (empty_) {
                state_ = continuate(0.0, std::arg(input)); // choose principal solution
                empty_ = false;
            } else {
                state_ = continuate(state_, std::arg(input) );
            }

            sqrt_ = std::polar(std::sqrt(std::abs(input)), 0.5* state_);

            return sqrt_;
        }

        /**
        * \brief Retrieve the stored reference solution.
        */
        std::complex<T> operator()() const
        {
            return sqrt_;
        }
        
        /**
        * @brief getter for state state
        * @return state_[-pi,pi]
        */
        T get_state(void) const
        {
            return state_;
        }
        
    };

}








