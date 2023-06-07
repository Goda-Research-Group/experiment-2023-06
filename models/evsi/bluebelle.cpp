
namespace EVSI {

Eigen::MatrixXd Bluebelle::_zRvs(int N, std::mt19937& mt) {
    Eigen::MatrixXd z(N, 1);
    static std::normal_distribution<> zDist(0, 1);
    for (int ni = 0; ni < N; ni++) {
        z(ni, 0) = zDist(mt);
    }
    return z;
}

double Bluebelle::expit(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Bluebelle::logit(double p) {
    return std::log(p / (1.0 - p));
}

Eigen::MatrixXd Bluebelle::NB(const Eigen::MatrixXd& theta) const {
    Eigen::MatrixXd f(4, 1);
    const std::vector<double> dressingCosts = {0, 5.25, 13.86, 21.39};
    const double lambda = 20000;
    const double SSIQARYloss = 0.12;

    const double DeltaE = theta(0, 0);
    const double DeltaG = theta(1, 0);
    const double DeltaA = theta(2, 0);
    const double pssiS = theta(3, 0);
    const double SSIcost = theta(4, 0);

    const std::vector<double> pssis = {
        expit(logit(pssiS) + DeltaE),
        pssiS,
        expit(logit(pssiS) + DeltaG),
        expit(logit(pssiS) + DeltaA),
    };

    for (int d = 0; d < 4; d++) {
        f(d, 0) =
            -dressingCosts[d] - pssis[d] * (SSIcost + lambda * SSIQARYloss);
    }

    return f;
}

int Bluebelle::dSize() const {
    return 4;
}

Eigen::MatrixXd Bluebelle::thetaRvs(std::mt19937& mt) const {
    counter++;

    const Eigen::MatrixXd Deltas = Sigma.llt().matrixL() * _zRvs(3, mt) + mu;
    static std::normal_distribution<> zDist(0, 1);
    const double pssiS = 0.001772214 * zDist(mt) + 0.137984898;
    const double SSIcost = std::exp(0.163148238 * zDist(mt) + 8.972237608);

    Eigen::MatrixXd theta(5, 1);
    theta.block(0, 0, 3, 1) = Deltas;
    theta(3, 0) = pssiS;
    theta(4, 0) = SSIcost;

    return theta;
}

Eigen::MatrixXd Bluebelle::phiRvs(const Eigen::MatrixXd& theta,
                                  std::mt19937& mt) const {
    counter++;
    return V.llt().matrixL() * _zRvs(3, mt) + theta.block(0, 0, 3, 1);
}

int Bluebelle::phiSize() const {
    return 3;
}

int Bluebelle::thetaSize() const {
    return 5;
}

Eigen::MatrixXd Bluebelle::getTheoreticalEVPI() const {
    return Eigen::MatrixXd::Constant(1, 1, 173.30);
}

Eigen::MatrixXd Bluebelle::getTheoreticalEVSI() const {
    double result;
    if (Ne == 25 && Ns == 25 && Ng == 25 && Na == 25) {
        result = 30.6;
    } else if (Ne == 50 && Ns == 50 && Ng == 50 && Na == 50) {
        result = 54.0;
    } else if (Ne == 75 && Ns == 75 && Ng == 75 && Na == 75) {
        result = 69.5;
    } else if (Ne == 100 && Ns == 100 && Ng == 100 && Na == 100) {
        result = 80.7;
    } else if (Ne == 125 && Ns == 125 && Ng == 125 && Na == 125) {
        result = 89.3;
    } else if (Ne == 250 && Ns == 250 && Ng == 250 && Na == 250) {
        result = 114.4;
    } else if (Ne == 375 && Ns == 375 && Ng == 375 && Na == 375) {
        result = 127.1;
    } else if (Ne == 500 && Ns == 500 && Ng == 500 && Na == 500) {
        result = 135.0;
    } else if (Ne == 625 && Ns == 625 && Ng == 625 && Na == 625) {
        result = 140.5;
    } else if (Ne == 750 && Ns == 750 && Ng == 750 && Na == 750) {
        result = 144.6;
    } else if (Ne == 875 && Ns == 875 && Ng == 875 && Na == 875) {
        result = 147.7;
    } else if (Ne == 1000 && Ns == 1000 && Ng == 1000 && Na == 1000) {
        result = 150.2;
    } else if (Ne == 1125 && Ns == 1125 && Ng == 1125 && Na == 1125) {
        result = 152.3;
    } else if (Ne == 1250 && Ns == 1250 && Ng == 1250 && Na == 1250) {
        result = 154.0;
    } else if (Ne == 50 && Ns == 50 && Ng == 50 && Na < 1) {
        result = 173.30 - 134.5388;
    } else if (Ne == 100 && Ns == 100 && Ng == 100 && Na < 1) {
        result = 173.30 - 112.3182;
    } else if (Ne == 150 && Ns == 150 && Ng == 150 && Na < 1) {
        result = 173.30 - 99.3786;
    } else if (Ne == 200 && Ns == 200 && Ng == 200 && Na < 1) {
        result = 173.30 - 90.5592;
    } else if (Ne == 300 && Ns == 300 && Ng == 300 && Na < 1) {
        result = 173.30 - 79.6541;
    } else if (Ne == 400 && Ns == 400 && Ng == 400 && Na < 1) {
        result = 173.30 - 72.6756;
    } else if (Ne == 500 && Ns == 500 && Ng == 500 && Na < 1) {
        result = 173.30 - 67.9556;
    } else if (Ne == 600 && Ns == 600 && Ng == 600 && Na < 1) {
        result = 173.30 - 64.5477;
    } else if (Ne == 700 && Ns == 700 && Ng == 700 && Na < 1) {
        result = 173.30 - 61.9662;
    } else if (Ne == 800 && Ns == 800 && Ng == 800 && Na < 1) {
        result = 173.30 - 59.7426;
    } else if (Ne == 900 && Ns == 900 && Ng == 900 && Na < 1) {
        result = 173.30 - 58.0293;
    } else if (Ne == 1000 && Ns == 1000 && Ng == 1000 && Na < 1) {
        result = 173.30 - 56.5608;
    } else if (Ne == 1100 && Ns == 1100 && Ng == 1100 && Na < 1) {
        result = 173.30 - 55.4932;
    } else if (Ne == 1200 && Ns == 1200 && Ng == 1200 && Na < 1) {
        result = 173.30 - 54.474;
    } else if (Ne == 1300 && Ns == 1300 && Ng == 1300 && Na < 1) {
        result = 173.30 - 53.4894;
    } else if (Ne == 1400 && Ns == 1400 && Ng == 1400 && Na < 1) {
        result = 173.30 - 52.814;
    } else if (Ne == 1500 && Ns == 1500 && Ng == 1500 && Na < 1) {
        result = 173.30 - 52.0089;
    } else if (Ne == 1600 && Ns == 1600 && Ng == 1600 && Na < 1) {
        result = 173.30 - 51.3536;
    } else if (Ne == 100 && Ns == 100 && Ng == 50 && Na < 1) {
        result = 173.30 - 127.1557;
    } else if (Ne == 200 && Ns == 200 && Ng == 100 && Na < 1) {
        result = 173.30 - 103.9948;
    } else if (Ne == 400 && Ns == 400 && Ng == 200 && Na < 1) {
        result = 173.30 - 82.5684;
    } else if (Ne == 600 && Ns == 600 && Ng == 300 && Na < 1) {
        result = 173.30 - 72.1989;
    } else if (Ne == 800 && Ns == 800 && Ng == 400 && Na < 1) {
        result = 173.30 - 66.0556;
    } else if (Ne == 1000 && Ns == 1000 && Ng == 500 && Na < 1) {
        result = 173.30 - 61.8838;
    } else if (Ne == 1200 && Ns == 1200 && Ng == 600 && Na < 1) {
        result = 173.30 - 58.9038;
    } else if (Ne == 1400 && Ns == 1400 && Ng == 700 && Na < 1) {
        result = 173.30 - 56.7535;
    } else if (Ne == 1600 && Ns == 1600 && Ng == 800 && Na < 1) {
        result = 173.30 - 54.9811;
    } else if (Ne == 1800 && Ns == 1800 && Ng == 900 && Na < 1) {
        result = 173.30 - 53.5607;
    } else if (Ne == 2000 && Ns == 2000 && Ng == 1000 && Na < 1) {
        result = 173.30 - 52.5085;
    } else {
        assert(false);
    }
    return Eigen::MatrixXd::Constant(1, 1, result);
}

}  // namespace EVSI
