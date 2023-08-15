#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

#include <Eigen/Core>


#include "types.hpp"
#include "hawp_commons.hpp"
#include "hawp_paramset.hpp"
#include "multi_index.hpp"
#include "shape.hpp"
#include "innerproducts/gauss_hermite_qr.hpp"
#include "innerproducts/tensor_product_qr.hpp"
#include "innerproducts/genz_keister_qr.hpp"
#include "innerproducts/homogeneous_inner_product.hpp"
#include "innerproducts/inhomogeneous_inner_product.hpp"
#include "innerproducts/vector_inner_product.hpp"
#include "potentials/potentials.hpp"
#include "ScalarHaWp_Propagator.hpp"
#include "Vector_Propagator.hpp"
#include "Observables.hpp"




void test_torsion_5D(){

    std::cout<<"5D torsion potential case:\n";
    std::cout <<   "------------------------------------------------\n";

    const int D = 5;   // the dimension of the system 
    const int K = 4;   // the sparsity of the basis shape 

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=std::sqrt(0.01);
    RMatrix<D,1> q;
    q <<  1.0, 1.0, 1.0, 1.0, 1.0;
    RMatrix<D,1> p;
    p <<  0.0, 0.0, 0.0, 0.0, 0.0;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs = Coefficients::Zero(shape_basis.size(),1);
    coeffs(0)=complex_t(1.0,0.0);
    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet(eps, param_set, coeffs, shape_basis);

    using QR = innerproducts::GaussHermiteQR<8>;
    using TQR = innerproducts::TensorProductQR<QR,QR,QR,QR,QR>;

    using PES=potentials::Torsion_XD<D>;     // scalar potential energy surface 
    RMatrix<D, D> mass_inv(D,D);
    mass_inv <<  1.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 1.0;

    PES pes=PES(mass_inv);  

    potentials::MatrixPotential1S<D, PES>  matrix1s(pes);
    using propagator=propagators::ScalarHaWp_Propagator<D,Wavepacket,TQR, PES>;
    propagator prop(matrix1s,packet);

    double dt=0.005;
    int    Nt=int(10/dt);
    int NtSave=4;

    std::ofstream Filepara("param_set.dat");        //output the parameter set 
    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;
    double q_norm=prop.packet().parameters().q().norm();   //norm of q
    double p_norm=prop.packet().parameters().p().norm();   //norm of p
    double detQ_abs=std::abs(prop.packet().parameters().Q().determinant());   //abs of the determinant of Q
    double detP_abs=std::abs(prop.packet().parameters().P().determinant());   //abs of the determinant of P
    double S_real=prop.packet().parameters().S().real();                      //the real value of the S

    double pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
    double kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
    complex_t corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
    std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){

        prop.propagate(dt);

        tt=it*dt;

        if(it%NtSave==0){

            q_norm=prop.packet().parameters().q().norm();   //norm of q
            p_norm=prop.packet().parameters().p().norm();   //norm of p
            detQ_abs=std::abs(prop.packet().parameters().Q().determinant());   //abs of the determinant of Q
            detP_abs=std::abs(prop.packet().parameters().P().determinant());   //abs of the determinant of P
            S_real=prop.packet().parameters().S().real();                      //the real value of the S
            pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
            kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
            corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 
    
            std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
            std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

            Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

    TQR::clear_cache();

}


void test_torsion_2D(){

    std::cout<<"2D torsion potential case:\n";
    std::cout <<   "------------------------------------------------\n";

    const int D = 2;
    const int K = 16;

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=std::sqrt(0.1);
    RMatrix<D,1> q;
    q <<  1.0, 0.0;
    RMatrix<D,1> p;
    p <<  0.0, 0.0;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs = Coefficients::Zero(shape_basis.size(),1);
    coeffs(0)=complex_t(1.0,0.0);
    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet(eps, param_set, coeffs, shape_basis);

    using QR = innerproducts::GaussHermiteQR<20>;
    using TQR = innerproducts::TensorProductQR<QR,QR>;

    using PES=potentials::Torsion_XD<D>;     // scalar potential energy surface 
    RMatrix<D, D> mass_inv(D,D);
    mass_inv <<  1.0, 0.0,
                 0.0, 1.0;

    PES pes=PES(mass_inv);  

    potentials::MatrixPotential1S<D, PES>  matrix1s(pes);
    using propagator=propagators::ScalarHaWp_Propagator<D,Wavepacket,TQR, PES>;
    propagator prop(matrix1s,packet);

    double dt=0.002;
    int    Nt=int(20/dt);
    int NtSave=4;

    std::ofstream Filepara("param_set.dat");        //output the parameter set 
    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;
    double q_norm=prop.packet().parameters().q().norm();   //norm of q
    double p_norm=prop.packet().parameters().p().norm();   //norm of p
    double detQ_abs=std::abs(prop.packet().parameters().Q().determinant());   //abs of the determinant of Q
    double detP_abs=std::abs(prop.packet().parameters().P().determinant());   //abs of the determinant of P
    double S_real=prop.packet().parameters().S().real();                      //the real value of the S
    double pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
    double kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
    complex_t corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
    std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){

        prop.propagate(dt);

        tt=it*dt;

        if(it%NtSave==0){

            q_norm=prop.packet().parameters().q().norm();   //norm of q
            p_norm=prop.packet().parameters().p().norm();   //norm of p
            detQ_abs=std::abs(prop.packet().parameters().Q().determinant());   //abs of the determinant of Q
            detP_abs=std::abs(prop.packet().parameters().P().determinant());   //abs of the determinant of P
            S_real=prop.packet().parameters().S().real();                      //the real value of the S
            pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
            kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
            corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 

    
            std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
            std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

            Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

    TQR::clear_cache();

}

void test_harmonic_2D(){

    std::cout<<"2D harmonic oscillator V(x,y): 0.5*(0.5*x^2+0.5*y^2):\n";
    std::cout <<   "------------------------------------------------\n";

    const int D = 2;
  //  const int K = 15;

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(1);   // set the sparsity to 1 since Hagedorn wavepacket is exact for the 2D harmonic oscillator 
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=0.1;
    RMatrix<D,1> q;
    q <<  1.8, 1.2;
    RMatrix<D,1> p;
    p <<  0.6, 0.8;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs = Coefficients::Zero(shape_basis.size(),1);
    coeffs(0)=complex_t(1.0,0.0);
    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>; 
    Wavepacket packet(eps, param_set, coeffs, shape_basis);

    using QR = innerproducts::GaussHermiteQR<10>;
    using TQR = innerproducts::TensorProductQR<QR,QR>;

    using PES=potentials::Harmonic_2D<D>;     // 2D harmonic oscillator 
    RMatrix<D, D> mass_inv(D,D);
    mass_inv <<  1.0, 0.0,
                 0.0, 1.0;

    PES pes=PES(0.5, 0.5, mass_inv);  //V(x,y): 0.5*(0.5*x^2+0.5*y^2)    

    potentials::MatrixPotential1S<D, PES>  matrix1s(pes);
    using propagator=propagators::ScalarHaWp_Propagator<D,Wavepacket,TQR, PES>;
    propagator prop(matrix1s,packet);

    double dt=0.01;
    int    Nt=int(12/dt);
    int NtSave=4;

    std::ofstream Filepara("param_set.dat");        //output the parameter set 
    std::ofstream FileObserve("Observables.dat");   //output the observable 
    std::ofstream FileTraj("Trajectory.dat");       //Trajectory of q, p, detQ, detP


    double tt=0.0;
    double q_norm=prop.packet().parameters().q().norm();   //norm of q
    double p_norm=prop.packet().parameters().p().norm();   //norm of p
    double detQ_abs=std::abs(prop.packet().parameters().Q().determinant());   //abs of the determinant of Q
    double detP_abs=std::abs(prop.packet().parameters().P().determinant());   //abs of the determinant of P

    RMatrix<D,1> q_t=prop.packet().parameters().q();       // the value of q
    RMatrix<D,1> p_t=prop.packet().parameters().p();       // the value of p
    complex_t detQ=prop.packet().parameters().Q().determinant();   // the determinant of Q: detQ
    complex_t detP=prop.packet().parameters().P().determinant();   // the determinant of P: detP

    double S_real=prop.packet().parameters().S().real();                      //the real value of the S
    double pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
    double kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
    complex_t corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
    std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";

    std::cout<<"q: "<<"\n"; 
    std::cout<<q_t(0)<<" "<<q_t(1)<<"\n";

    std::cout<<"p: "<<"\n";
    std::cout<<p_t(0)<<" "<<p_t(1)<<"\n";

    std::cout<<"detQ "<<" detP "<<"\n";
    std::cout<<detQ<<" "<<detP<<"\n";

    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";
    FileTraj<<tt<<" "<<q_t(0)<<" "<<q_t(1)<<" "<<p_t(0)<<" "<<p_t(1)<<" "<<detQ.real()<<" "<<detQ.imag()<<" "<<detP.real()<<" "<<detP.imag()<<"\n";



    
    for(int it=1; it<=Nt; it++){

        //prop.propagate(dt);
        prop.propagate_order4(dt);

        tt=it*dt;

        if(it%NtSave==0){

            q_norm=prop.packet().parameters().q().norm();   //norm of q
            p_norm=prop.packet().parameters().p().norm();   //norm of p
            detQ_abs=std::abs(prop.packet().parameters().Q().determinant());   //abs of the determinant of Q
            detP_abs=std::abs(prop.packet().parameters().P().determinant());   //abs of the determinant of P

            q_t=prop.packet().parameters().q();       // the value of q
            p_t=prop.packet().parameters().p();       // the value of p
            detQ=prop.packet().parameters().Q().determinant();   // the determinant of Q: detQ
            detP=prop.packet().parameters().P().determinant();   // the determinant of P: detP

            S_real=prop.packet().parameters().S().real();                      //the real value of the S
            pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
            kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
            corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 


            std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
            std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";

            std::cout<<"q: "<<"\n"; 
            std::cout<<q_t(0)<<" "<<q_t(1)<<"\n";

            std::cout<<"p: "<<"\n";
            std::cout<<p_t(0)<<" "<<p_t(1)<<"\n";

            std::cout<<"detQ "<<" detP "<<"\n";
            std::cout<<detQ<<" "<<detP<<"\n";

            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

            Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";
            FileTraj<<tt<<" "<<q_t(0)<<" "<<q_t(1)<<" "<<p_t(0)<<" "<<p_t(1)<<" "<<detQ.real()<<" "<<detQ.imag()<<" "<<detP.real()<<" "<<detP.imag()<<"\n";

        }

    }

    TQR::clear_cache();

}


void test_morse_1D(){

    std::cout<<"1D morse potential V(x)=D_e(e^{-2a(x-x_0)}-2e^{x-x_0}):\n";
    std::cout <<   "------------------------------------------------\n";

    const int D = 1;
    const int K = 50;

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=std::sqrt(1.0);
    RMatrix<D,1> q;
    q <<  5.0;
    RMatrix<D,1> p;
    p <<  0.0;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs = Coefficients::Zero(shape_basis.size(),1);
    coeffs(0)=complex_t(1.0,0.0);
    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet(eps, param_set, coeffs, shape_basis);

    using QR = innerproducts::GaussHermiteQR<15>;

    using PES=potentials::Morse_zero_1D<D>;     // scalar potential energy surface 
    RMatrix<D, D> mass_inv(D,D);
    mass_inv <<  1.0;

    PES pes=PES(10.25, 0.2209, 0.0, mass_inv);  

    potentials::MatrixPotential1S<D, PES>  matrix1s(pes);
    using propagator=propagators::ScalarHaWp_Propagator<D,Wavepacket,QR, PES>;
    propagator prop(matrix1s,packet);

    double dt=0.005;
    int    Nt=int(20/dt);
    int NtSave=4;

    std::ofstream Filepara("param_set.dat");        //output the parameter set 
    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;
    RMatrix<D,1> q_t=prop.packet().parameters().q();       // the value of q
    RMatrix<D,1> p_t=prop.packet().parameters().p();       // the value of p 
    double pot=observables::potential_energy<D, Wavepacket, QR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
    double kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
    complex_t corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, QR>(packet, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" q "<<" p "<<"\n";
    std::cout<<tt<<" "<<q_t(0)<<" "<<p_t(0)<<"\n";
    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    Filepara<<tt<<" "<<q(0)<<" "<<p(0)<<"\n";
    FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){

        prop.propagate(dt);

        tt=it*dt;

        if(it%NtSave==0){

            q_t=prop.packet().parameters().q();       // the value of q
            p_t=prop.packet().parameters().p();       // the value of p 
            pot=observables::potential_energy<D, Wavepacket, QR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
            kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
            corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, QR>(packet, prop.packet());   //autocorrelation function 


            std::cout<<"time "<<" q "<<" p "<<"\n";
            std::cout<<tt<<" "<<q_t(0)<<" "<<p_t(0)<<"\n";
            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

            Filepara<<tt<<" "<<q(0)<<" "<<p(0)<<"\n";
            FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

}


void test_avoided_crossing_2D(){

    std::cout<<"2D avoided crossing case:  homogenous propagation \n";
    std::cout <<   "------------------------------------------------\n";

    const dim_t N = 2;    // the energy level 
    const dim_t D = 2;    // the dimension of the system 
    const int K = 10;   // the cut-off for the hyperbolic cut basis shape 
    const real_t delta=0.01;  //energy gap for the avoided crossing pes
    const dim_t leading_order=0;   //the upper surface 

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=0.01;
    RMatrix<D,1> q;
    q <<  -3.0, 0.0;
    RMatrix<D,1> p;
    p <<  0.5, 0.0;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs0 = Coefficients::Zero(shape_basis.size(), 1);   //upper surface 
    coeffs0(0)= complex_t(1.0,0.0); 
    Coefficients coeffs1 = Coefficients::Zero(shape_basis.size(),1);    //lower surface 



    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet0(eps, param_set, coeffs0, shape_basis);   //wavepackets in the upper surface 
    Wavepacket packet1(eps, param_set, coeffs1, shape_basis);   //wavepackets in the lower surface 
    std::tuple<Wavepacket, Wavepacket> components(packet0, packet1);  //component for the vectorized Hagedorn wavepacket 
    using VectorWavepacket=wavepackets::VectorHaWp2<D, Wavepacket, Wavepacket>;   
    VectorWavepacket VecPacket(eps, components);  //the vectorized Hagedorn wavepacket 

    
    using QR = innerproducts::GaussHermiteQR<10>;
    using TQR = innerproducts::TensorProductQR<QR,QR>;

    using PES=potentials::delta_gap_2D_2N<D, N>;   //matrix potential energy surface 
    RMatrix<D, D> mass_inv(D,D);
    mass_inv <<  1.0, 0.0,
                 0.0, 1.0;
    PES pes=PES(leading_order, delta, mass_inv);  

    potentials::MatrixPotentialMS<D, PES>  matrixms(pes);


    using propagator=propagators::VectorHaWp_Propagator<D,VectorWavepacket,TQR, PES>;
    propagator prop(matrixms,VecPacket);

    double dt=0.01;
    int    Nt=int(10/dt);
    int NtSave=4;

    std::ofstream Filepara("param_set.dat");        //output the parameter set 
    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;

    /*
    std::cout<<"size::"<<prop.packet().size()<<std::endl;
    std::vector<dim_t> offsets=prop.packet().offset();
    std::cout<<"offsets:"<<offsets[0]<<" "<<offsets[1]<<"\n";*/

    double q_norm=prop.packet().component(0).parameters().q().norm();   //norm of q
    double p_norm=prop.packet().component(0).parameters().p().norm();   //norm of p
    double detQ_abs=std::abs(prop.packet().component(0).parameters().Q().determinant());   //abs of the determinant of Q
    double detP_abs=std::abs(prop.packet().component(0).parameters().P().determinant());   //abs of the determinant of P
    double S_real=prop.packet().component(0).parameters().S().real();                      //the real value of the S
    double pot=observables::potential_energy_vec<D, VectorWavepacket, TQR, potentials::MatrixPotentialMS<D, PES>>(prop.packet(), prop.matrixms());  //the potential energy 
    double kin=observables::kinetic_energy_vec<D, VectorWavepacket>(prop.packet(),prop.matrixms().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm_vec<D, VectorWavepacket>(prop.packet());           //norm of the wavepacket 
    RMatrix<Eigen::Dynamic, 1>  pop=observables::pop_vec<D, VectorWavepacket>(prop.packet());  //population 
    complex_t corr_packet=observables::auto_corr_vec<D, VectorWavepacket, VectorWavepacket, TQR>(VecPacket, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
    std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
    FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){
        std::cout<<"it: "<<it<<"\n";
        prop.propagate_homo(dt);
        //prop.propagate_inhomo(dt);

        tt=it*dt;

        if(it%NtSave==0){

            q_norm=prop.packet().component(0).parameters().q().norm();   //norm of q
            p_norm=prop.packet().component(0).parameters().p().norm();   //norm of p
            detQ_abs=std::abs(prop.packet().component(0).parameters().Q().determinant());   //abs of the determinant of Q
            detP_abs=std::abs(prop.packet().component(0).parameters().P().determinant());   //abs of the determinant of P
            S_real=prop.packet().component(0).parameters().S().real();                      //the real value of the S
            pot=observables::potential_energy_vec<D, VectorWavepacket, TQR, potentials::MatrixPotentialMS<D, PES>>(prop.packet(), prop.matrixms());  //the potential energy 
            kin=observables::kinetic_energy_vec<D, VectorWavepacket>(prop.packet(),prop.matrixms().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm_vec<D, VectorWavepacket>(prop.packet());           //norm of the wavepacket 
            pop=observables::pop_vec<D, VectorWavepacket>(prop.packet());  //population 
            corr_packet=observables::auto_corr_vec<D, VectorWavepacket, VectorWavepacket, TQR>(VecPacket, prop.packet());   //autocorrelation function 


            std::cout<<"time "<<" |q| "<<" |p| "<<" |detQ| "<<" |det P| "<< " S "<<"\n";
            std::cout<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

            Filepara<<tt<<" "<<q_norm<<" "<<p_norm<<" "<<detQ_abs<<" "<<detP_abs<<" "<<S_real<<"\n";
            FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

    TQR::clear_cache();  

}


void test_henon_heiles_4D(){

    std::cout<<"4D henon-heiles case:\n";
    std::cout <<   "------------------------------------------------\n";

    const int D = 4;   // the dimension of the system 
    const int K = 1;   // the sparsity of the basis shape 
    const real_t lambda=0.11803;    //the mode-mode coupling strength for the henon-heiles potential

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=std::sqrt(1.0);
    RMatrix<D,1> q;
    q <<  2.0, 2.0, 2.0, 2.0;
    RMatrix<D,1> p;
    p <<  0.0, 0.0, 0.0, 0.0;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs = Coefficients::Zero(shape_basis.size(),1);
    coeffs(0)=complex_t(1.0,0.0);
    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet(eps, param_set, coeffs, shape_basis);

    //using QR = innerproducts::GaussHermiteQR<10>;
    //using TQR = innerproducts::TensorProductQR<QR,QR,QR,QR>;
    using TQR=innerproducts::GenzKeisterQR<D, 20>;

    using PES=potentials::Henon_Heiles<D>;     // scalar potential energy surface 
    RMatrix<D, D> mass_inv(D,D);
    mass_inv <<  1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0, 
                 0.0, 0.0, 1.0, 0.0, 
                 0.0, 0.0, 0.0, 1.0;

    PES pes=PES(lambda, mass_inv);  

    potentials::MatrixPotential1S<D, PES>  matrix1s(pes);
    using propagator=propagators::ScalarHaWp_Propagator<D,Wavepacket,TQR, PES>;
    propagator prop(matrix1s,packet);

    double dt=0.002;
    int    Nt=int(30/dt);
    int NtSave=1;

    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;

    double pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
    double kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
    complex_t corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 

    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){

        prop.propagate(dt);

        tt=it*dt;

        if(it%NtSave==0){

            pot=observables::potential_energy<D, Wavepacket, TQR, potentials::MatrixPotential1S<D, PES>>(prop.packet(), prop.matrix1s());  //the potential energy 
            kin=observables::kinetic_energy<D, Wavepacket>(prop.packet(),prop.matrix1s().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm<D, Wavepacket>(prop.packet());           //norm of the wavepacket 
            corr_packet=observables::auto_corr<D, Wavepacket, Wavepacket, TQR>(packet, prop.packet());   //autocorrelation function 

            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";
            FileObserve<<tt<<" "<<pot<<" "<<kin<<" "<<pot+kin<<" "<<norm_packet<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

    TQR::clear_cache();

}



void test_pyrazine_4D(){

    std::cout<<"4D pyrazine:  homogenous propagation \n";
    std::cout <<   "------------------------------------------------\n";

    const dim_t N = 2;    // the energy level 
    const dim_t D = 4;    // the dimension of the system 
    const dim_t K = 4;   // the cut-off for the hyperbolic cut basis shape 
    const dim_t leading_order=1;   //S_2 diabatic state 

    const real_t time_scale=55.887401309781374;   //the scaled time (fs)
    const real_t hbar=27.211386245988;            // hatree to eV 

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=sqrt(0.002719449840998);
    RMatrix<D,1> q;
    q <<  0.0, 0.0, 0.0, 0.0;
    RMatrix<D,1> p;
    p <<  0.0, 0.0, 0.0, 0.0;

    RMatrix<D, D> Qdiag(D,D);

    Qdiag <<  1.0/std::sqrt(1.264864864864865), 0.0, 0.0, 0.0,
                 0.0, 1.0/std::sqrt(1.0), 0.0, 0.0, 
                 0.0, 0.0, 1.0/std::sqrt(1.720270270270270), 0.0, 
                 0.0, 0.0, 0.0, 1.0/std::sqrt(2.118918918918919);

    CMatrix<D,D> Q=complex_t(1.0, 0.0)*Qdiag;

    RMatrix<D, D> Pdiag(D,D);

    Pdiag <<  std::sqrt(1.264864864864865), 0.0, 0.0, 0.0,
                 0.0, std::sqrt(1.0), 0.0, 0.0, 
                 0.0, 0.0, std::sqrt(1.720270270270270), 0.0, 
                 0.0, 0.0, 0.0, std::sqrt(2.118918918918919);

    CMatrix<D,D> P=complex_t(0,1) * Pdiag;
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs0 = Coefficients::Zero(shape_basis.size(),1);   //diabatic S1 state  
    Coefficients coeffs1 = Coefficients::Zero(shape_basis.size(),1);    //diabatic S2 state 
    coeffs1(0)= complex_t(1.0,0.0);



    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet0(eps, param_set, coeffs0, shape_basis);   //wavepackets in the diabatic S1 state 
    Wavepacket packet1(eps, param_set, coeffs1, shape_basis);   //wavepackets in the diabatic S2 state 
    std::tuple<Wavepacket, Wavepacket> components(packet0, packet1);  //component for the vectorized Hagedorn wavepacket 
    using VectorWavepacket=wavepackets::VectorHaWp2<D, Wavepacket, Wavepacket>;   
    VectorWavepacket VecPacket(eps, components);  //the vectorized Hagedorn wavepacket 

    
    using QR = innerproducts::GaussHermiteQR<8>;
    using TQR = innerproducts::TensorProductQR<QR,QR,QR,QR>;
    //using TQR=innerproducts::GenzKeisterQR<D, 15>;

    using PES=potentials::pyrazine_4D<D, N>;   //matrix potential energy surface 
    RMatrix<D, D> mass_inv(D,D);

    mass_inv <<  1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0, 
                 0.0, 0.0, 1.0, 0.0, 
                 0.0, 0.0, 0.0, 1.0;

    PES pes=PES(leading_order, mass_inv);  

    potentials::MatrixPotentialMS<D, PES>  matrixms(pes);


    using propagator=propagators::VectorHaWp_Propagator<D,VectorWavepacket,TQR, PES>;
    propagator prop(matrixms,VecPacket);

    double dt=0.01/time_scale;     // should multiple time_scale 
    int    Nt=int(120/time_scale/dt);  
    int NtSave=5;

    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;


    double pot=observables::potential_energy_vec<D, VectorWavepacket, TQR, potentials::MatrixPotentialMS<D, PES>>(prop.packet(), prop.matrixms());  //the potential energy 
    double kin=observables::kinetic_energy_vec<D, VectorWavepacket>(prop.packet(),prop.matrixms().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm_vec<D, VectorWavepacket>(prop.packet());           //norm of the wavepacket 
    RMatrix<Eigen::Dynamic, 1>  pop=observables::pop_vec<D, VectorWavepacket>(prop.packet());  //population 
    complex_t corr_packet=observables::auto_corr_vec<D, VectorWavepacket, VectorWavepacket, TQR>(VecPacket, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    FileObserve<<tt<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){
        std::cout<<"it: "<<it<<"\n";
        prop.propagate_homo(dt);
        //prop.propagate_inhomo(dt);

        tt=it*dt;

        if(it%NtSave==0){

            pot=observables::potential_energy_vec<D, VectorWavepacket, TQR, potentials::MatrixPotentialMS<D, PES>>(prop.packet(), prop.matrixms());  //the potential energy 
            kin=observables::kinetic_energy_vec<D, VectorWavepacket>(prop.packet(),prop.matrixms().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm_vec<D, VectorWavepacket>(prop.packet());           //norm of the wavepacket 
            pop=observables::pop_vec<D, VectorWavepacket>(prop.packet());  //population 
            corr_packet=observables::auto_corr_vec<D, VectorWavepacket, VectorWavepacket, TQR>(VecPacket, prop.packet());   //autocorrelation function 


            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt*time_scale<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


            FileObserve<<tt*time_scale<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

    TQR::clear_cache();  
    FileObserve.close();

}

void test_pyrazine_4D_noscale(){

    std::cout<<"4D pyrazine:  homogenous propagation \n";
    std::cout <<   "------------------------------------------------\n";

    const dim_t N = 2;    // the energy level 
    const dim_t D = 4;    // the dimension of the system 
    const dim_t K = 4;   // the cut-off for the hyperbolic cut basis shape 
    const dim_t leading_order=1;   //S_2 diabatic state 

    const real_t hbar=0.658229;   //Planck constant (eV.fs)

    using CMatrix1X = CMatrix<1, Eigen::Dynamic>;
    using RMatrixD1 = RMatrix<D, 1>;
    using CMatrixDX = CMatrix<D, Eigen::Dynamic>;

    using Shape=shapes::HyperbolicCutShape<D>;
    Shape shape_basis=Shape(K);
    std::cout<<"basis size: "<<shape_basis.size()<<std::endl;

    const real_t eps=sqrt(1.0);
    RMatrix<D,1> q;
    q <<  0.0, 0.0, 0.0, 0.0;
    RMatrix<D,1> p;
    p <<  0.0, 0.0, 0.0, 0.0;
    CMatrix<D,D> Q=CMatrix<D,D>::Identity();
    CMatrix<D,D> P=complex_t(0,1) * CMatrix<D,D>::Identity();
    complex_t  S=complex_t(0.0, 0.0);

    wavepackets::HaWpParamSet<D> param_set(q,p,Q,P,S);
    std::cout << param_set << std::endl;

    Coefficients coeffs0 = Coefficients::Zero(shape_basis.size(),1);   //diabatic S1 state  
    Coefficients coeffs1 = Coefficients::Zero(shape_basis.size(),1);    //diabatic S2 state 
    coeffs1(0)= complex_t(1.0,0.0);



    using Wavepacket=wavepackets::ScalarHaWp<D, Shape>;
    Wavepacket packet0(eps, param_set, coeffs0, shape_basis);   //wavepackets in the diabatic S1 state 
    Wavepacket packet1(eps, param_set, coeffs1, shape_basis);   //wavepackets in the diabatic S2 state 
    std::tuple<Wavepacket, Wavepacket> components(packet0, packet1);  //component for the vectorized Hagedorn wavepacket 
    using VectorWavepacket=wavepackets::VectorHaWp2<D, Wavepacket, Wavepacket>;   
    VectorWavepacket VecPacket(eps, components);  //the vectorized Hagedorn wavepacket 

    
    using QR = innerproducts::GaussHermiteQR<8>;
    using TQR = innerproducts::TensorProductQR<QR,QR,QR,QR>;
    //using TQR=innerproducts::GenzKeisterQR<D, 15>;

    using PES=potentials::pyrazine_4D_noscale<D, N>;   //matrix potential energy surface 
    RMatrix<D, D> mass_inv(D,D);

    mass_inv <<  0.0936, 0.0, 0.0, 0.0,
                 0.0, 0.0740, 0.0, 0.0, 
                 0.0, 0.0, 0.1273, 0.0, 
                 0.0, 0.0, 0.0, 0.1568;

    mass_inv=mass_inv/hbar;  //eV to fs-1

    PES pes=PES(leading_order, mass_inv);  

    potentials::MatrixPotentialMS<D, PES>  matrixms(pes);


    using propagator=propagators::VectorHaWp_Propagator<D,VectorWavepacket,TQR, PES>;
    propagator prop(matrixms,VecPacket);

    double dt=0.01;     // fs 
    int    Nt=int(120/dt);  
    int NtSave=5;

    std::ofstream FileObserve("Observables.dat");   //output the observable 


    double tt=0.0;


    double pot=observables::potential_energy_vec<D, VectorWavepacket, TQR, potentials::MatrixPotentialMS<D, PES>>(prop.packet(), prop.matrixms());  //the potential energy 
    double kin=observables::kinetic_energy_vec<D, VectorWavepacket>(prop.packet(),prop.matrixms().pes().mass_inv());    //the kinetic energy 
    double norm_packet=observables::norm_vec<D, VectorWavepacket>(prop.packet());           //norm of the wavepacket 
    RMatrix<Eigen::Dynamic, 1>  pop=observables::pop_vec<D, VectorWavepacket>(prop.packet());  //population 
    complex_t corr_packet=observables::auto_corr_vec<D, VectorWavepacket, VectorWavepacket, TQR>(VecPacket, prop.packet());   //autocorrelation function 


    std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
    std::cout<<tt<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

    FileObserve<<tt<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


    
    for(int it=1; it<=Nt; it++){
        std::cout<<"it: "<<it<<"\n";
        prop.propagate_homo(dt);
        //prop.propagate_inhomo(dt);

        tt=it*dt;

        if(it%NtSave==0){

            pot=observables::potential_energy_vec<D, VectorWavepacket, TQR, potentials::MatrixPotentialMS<D, PES>>(prop.packet(), prop.matrixms());  //the potential energy 
            kin=observables::kinetic_energy_vec<D, VectorWavepacket>(prop.packet(),prop.matrixms().pes().mass_inv());    //the kinetic energy 
            norm_packet=observables::norm_vec<D, VectorWavepacket>(prop.packet());           //norm of the wavepacket 
            pop=observables::pop_vec<D, VectorWavepacket>(prop.packet());  //population 
            corr_packet=observables::auto_corr_vec<D, VectorWavepacket, VectorWavepacket, TQR>(VecPacket, prop.packet());   //autocorrelation function 


            std::cout<<"time "<<" pot "<<" kin "<<" tot "<<" norm "<<" auto(real) "<<" auto(imag) "<<"\n";
            std::cout<<tt<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";


            FileObserve<<tt<<" "<<hbar*pot<<" "<<hbar*kin<<" "<<hbar*(pot+kin)<<" "<<norm_packet<<" "<<pop(0)<<" "<<pop(1)<<" "<<corr_packet.real()<<" "<<corr_packet.imag()<<"\n";

        }

    }

    TQR::clear_cache();  
    FileObserve.close();

}




int main()
{
    //test_torsion_5D();
    //test_torsion_2D();
    //test_harmonic_2D();
    //test_morse_1D();
    //test_avoided_crossing_2D();
    //test_henon_heiles_4D();
    //test_pyrazine_4D();
    test_pyrazine_4D_noscale();
    return 0;
}
