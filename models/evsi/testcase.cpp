
namespace EVSI {

Eigen::MatrixXd TestCase::NB(const Eigen::MatrixXd& theta) const {
    Eigen::MatrixXd result(2, 1);
    result(0, 0) = theta(0, 0);
    result(1, 0) = 1 - theta(0, 0);
    return result;
}

int TestCase::dSize() const {
    return 2;
}

Eigen::MatrixXd TestCase::thetaRvs(std::mt19937& mt) const {
    counter++;
    std::bernoulli_distribution dist(0.5);
    return Eigen::MatrixXd::Constant(1, 1, (dist(mt) ? 1 : 0));
}

Eigen::MatrixXd TestCase::phiRvs(const Eigen::MatrixXd& theta,
                                 std::mt19937& mt) const {
    counter++;
    std::bernoulli_distribution dist(p);
    std::uniform_real_distribution<> u(0.0, 1.0);
    Eigen::MatrixXd phi(phiSize(), 1);
    for (int m = 0; m < M; m++) {
        double z = dist(mt);
        if (theta(0, 0) > 0.5) {
            phi(m, 0) = (z ? u(mt) : -u(mt));
        } else {
            phi(m, 0) = (z ? -u(mt) : u(mt));
        }
    }
    return phi;
}

int TestCase::phiSize() const {
    return M;
}

int TestCase::thetaSize() const {
    return 1;
}

Eigen::MatrixXd TestCase::getTheoreticalEVPI() const {
    return Eigen::MatrixXd::Constant(1, 1, 0.5);
}

Eigen::MatrixXd TestCase::getTheoreticalEVSI() const {
    double result = 0;
    unsigned int ma = 0x01 << M;
    for (unsigned int mai = 0; mai < ma; mai++) {
        double P1 = 0.5, P2 = 0.5;
        for (int mj = 0; mj < M; mj++) {
            P1 *= ((mai & (0x01 << mj)) != 0 ? p : 1 - p);
            P2 *= ((mai & (0x01 << mj)) != 0 ? 1 - p : p);
        }
        result += std::max(P1, P2);
    }
    return Eigen::MatrixXd::Constant(1, 1, result - 0.5);
}

}  // namespace EVSI
