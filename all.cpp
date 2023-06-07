#ifndef _ALL_CPP
#define _ALL_CPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <random>
#include <sstream>
#include <vector>

namespace NE {

class Model {
   public:
    virtual std::string getName() const = 0;
    virtual Eigen::MatrixXd f(const Eigen::MatrixXd& gy) const = 0;
    virtual int fSize() const = 0;
    virtual int ySize() const = 0;
    virtual int xSize() const = 0;
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> yxRvs(
        std::mt19937& mt) const = 0;
    virtual Eigen::MatrixXd getTheoretical() const = 0;
};

class Estimator {
   public:
    virtual std::string getName() const = 0;
    virtual Eigen::MatrixXd estimate(int N,
                                     const NE::Model& model,
                                     std::mt19937& mt) const = 0;
};

class Proposed : public NE::Estimator {};

class ProposedSimple : public Proposed {
   public:
    std::string getName() const { return "ProposedSimple"; }
    Eigen::MatrixXd estimate(int m,
                             const NE::Model& model,
                             std::mt19937& mt) const;
};

class ProposedSparseGrid : public Proposed {
   public:
    std::string getName() const { return "ProposedSparseGrid"; }
    Eigen::MatrixXd estimate(int m,
                             const NE::Model& model,
                             std::mt19937& mt) const;
};

class GAM : public NE::Estimator {
   private:
    const std::string tmpPath;

   public:
    GAM(std::string tmpPath) : tmpPath{tmpPath} {}
    std::string getName() const { return "GAM"; }
    Eigen::MatrixXd estimate(int N,
                             const NE::Model& model,
                             std::mt19937& mt) const;
};

}  // namespace NE

namespace EVSI {

class Model : public NE::Model {
   protected:
    int& counter;

   public:
    Model(int& counter) : counter{counter} {}
    virtual std::string getName() const = 0;
    Eigen::MatrixXd f(const Eigen::MatrixXd& gy) const {
        Eigen::MatrixXd gd = gy.bottomRows(dSize());
        Eigen::MatrixXd gm = gy.topRows(1);
        return gm - gd.colwise().maxCoeff();
    }
    int fSize() const { return 1; }
    int ySize() const { return phiSize(); }
    int xSize() const { return dSize() + 1; }
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> yxRvs(std::mt19937& mt) const {
        auto theta = thetaRvs(mt);
        auto phi = phiRvs(theta, mt);
        auto gd = NB(theta);
        auto gm = gd.colwise().maxCoeff();
        Eigen::MatrixXd result(gm.rows() + gd.rows(), gm.cols());
        result.topRows(gm.rows()) = gm;
        result.bottomRows(gd.rows()) = gd;
        return {phi, result};
    }
    Eigen::MatrixXd getTheoretical() const {
        return getTheoreticalEVPI() - getTheoreticalEVSI();
    }

   private:
    virtual Eigen::MatrixXd NB(const Eigen::MatrixXd& theta) const = 0;
    virtual int dSize() const = 0;
    virtual Eigen::MatrixXd thetaRvs(std::mt19937& mt) const = 0;
    virtual Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                                   std::mt19937& mt) const = 0;
    virtual int thetaSize() const = 0;
    virtual int phiSize() const = 0;
    virtual Eigen::MatrixXd getTheoreticalEVPI() const = 0;
    virtual Eigen::MatrixXd getTheoreticalEVSI() const = 0;
};

class TestCase : public Model {
   private:
    const int M;
    const double p;
    Eigen::MatrixXd NB(const Eigen::MatrixXd& theta) const;
    int dSize() const;
    Eigen::MatrixXd thetaRvs(std::mt19937& mt) const;
    Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                           std::mt19937& mt) const;
    int thetaSize() const;
    int phiSize() const;

   public:
    std::string getName() const {
        std::ostringstream ost;
        ost << "TestCase_" << M << "_" << p;
        return ost.str();
    }
    TestCase(int& counter, int M, double p) : Model{counter}, M{M}, p{p} {}
    Eigen::MatrixXd getTheoreticalEVPI() const;
    Eigen::MatrixXd getTheoreticalEVSI() const;
};

class Ades : public Model {
   private:
    const int scenario;
    typedef struct {
        double L, Q_E, Q_SE, C_E, C_T, C_SE, P_C, P_SE, OR, P_T, lambda;
    } thetaType;
    static thetaType matrixToStruct(const Eigen::MatrixXd& thetaMatrix);
    static Eigen::MatrixXd structToMatrix(const thetaType& thetaStruct);
    static double sigmoid(double x);
    static double logit(double p);

    Eigen::MatrixXd NB(const Eigen::MatrixXd& theta) const;
    int dSize() const;
    Eigen::MatrixXd thetaRvs(std::mt19937& mt) const;
    Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                           std::mt19937& mt) const;
    int thetaSize() const;
    int phiSize() const;

   public:
    std::string getName() const {
        std::ostringstream ost;
        ost << "Ades_" << scenario;
        return ost.str();
    }
    Ades(int& counter, int scenario) : Model{counter}, scenario{scenario} {}
    Eigen::MatrixXd getTheoreticalEVPI() const;
    Eigen::MatrixXd getTheoreticalEVSI() const;
};

class Medical : public Model {
   private:
    const int scenario;
    typedef struct {
        double L, Q_E, Q_SE, C_E, C_SE, C_T_1, C_T_2, C_T_3, P_E_1, logOR_E_2,
            logOR_E_3, P_E_2, P_E_3, P_SE_1, P_SE_2, P_SE_3, lambda;
    } thetaType;
    static thetaType matrixToStruct(const Eigen::MatrixXd& thetaMatrix);
    static Eigen::MatrixXd structToMatrix(const thetaType& thetaStruct);
    static double sigmoid(double x);
    static double logit(double p);

    Eigen::MatrixXd NB(const Eigen::MatrixXd& theta) const;
    int dSize() const;
    Eigen::MatrixXd thetaRvs(std::mt19937& mt) const;
    Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                           std::mt19937& mt) const;
    int thetaSize() const;
    int phiSize() const;

   public:
    std::string getName() const {
        std::ostringstream ost;
        ost << "Medical_" << scenario;
        return ost.str();
    }
    Medical(int& counter, int scenario) : Model{counter}, scenario{scenario} {}
    Eigen::MatrixXd getTheoreticalEVPI() const;
    Eigen::MatrixXd getTheoreticalEVSI() const;
};

class Bluebelle : public Model {
   private:
    static constexpr double s = 3.7;
    const double Ne, Ns, Ng, Na;
    Eigen::MatrixXd mu, V, Sigma, Si, Vi, SVi;
    static double expit(double x);
    static double logit(double p);
    static Eigen::MatrixXd _zRvs(int N, std::mt19937& mt);

    Eigen::MatrixXd NB(const Eigen::MatrixXd& theta) const;
    int dSize() const;
    Eigen::MatrixXd thetaRvs(std::mt19937& mt) const;
    Eigen::MatrixXd phiRvs(const Eigen::MatrixXd& theta,
                           std::mt19937& mt) const;
    int thetaSize() const;
    int phiSize() const;

   public:
    std::string getName() const {
        std::ostringstream ost;
        ost << "Bluebelle_" << Ne << "_" << Ns << "_" << Ng << "_" << Na;
        return ost.str();
    }
    Bluebelle(int& counter, double Ne, double Ns, double Ng, double Na)
        : Model{counter}, Ne{Ne}, Ns{Ns}, Ng{Ng}, Na{Na} {
        mu = Eigen::MatrixXd(3, 1);
        mu << -0.05021921904305, -0.06629096195095, -0.178047360396;
        Sigma = Eigen::MatrixXd(3, 3);
        Sigma << 0.06546909, 0.06274766, 0.01781241, 0.06274766, 0.21792037,
            0.01727998, 0.01781241, 0.01727998, 0.04631648;
        V = Eigen::MatrixXd(3, 3);
        V << s * s * (Ne + Ns) / (Ne * Ns), s * s / Ns, s * s / Ns, s * s / Ns,
            s * s * (Ng + Ns) / (Ng * Ns), s * s / Ns, s * s / Ns, s * s / Ns,
            s * s * (Na + Ns) / (Na * Ns);
        Si = Sigma.inverse();
        Vi = V.inverse();
        SVi = (Si + Vi).inverse();
    }
    Eigen::MatrixXd getTheoreticalEVPI() const;
    Eigen::MatrixXd getTheoreticalEVSI() const;
};

}  // namespace EVSI

#include "methods/gam.cpp"
#include "methods/proposed.cpp"
#include "models/evsi/bluebelle.cpp"
#include "models/evsi/testcase.cpp"
#include "visualize.cpp"

#endif
