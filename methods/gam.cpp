
#include <fstream>
#include <iostream>

namespace NE {

Eigen::MatrixXd smoothing(const Eigen::MatrixXd& xs,
                          const Eigen::MatrixXd& ys,
                          std::string tmpPath) {
    int N = xs.cols();
    int J = xs.rows();
    int K = ys.rows();

    assert(K <= 10);

    std::string inputX = tmpPath + "/inputX.txt";
    std::ofstream ofsX(inputX);
    if (!ofsX) {
        assert(false);
    }
    ofsX << xs.transpose() << std::endl;

    std::string inputY = tmpPath + "/inputY.txt";
    std::ofstream ofsY(inputY);
    if (!ofsY) {
        assert(false);
    }
    ofsY << ys.transpose() << std::endl;

    int ret = system(("Rscript methods/gam.R " + tmpPath).c_str());

    Eigen::MatrixXd xTilde;
    if (ret != 0) {
        xTilde = Eigen::MatrixXd(J, N);
        for (int ji = 0; ji < N; ji++) {
            for (int ni = 0; ni < N; ni++) {
                xTilde(ji, ni) = std::nan("");
            }
        }
    } else {
        xTilde = Eigen::MatrixXd(J, N);
        std::string outputX = tmpPath + "/outputX.txt";
        std::ifstream ifs(outputX);
        if (!ifs) {
            assert(false);
        }
        for (int ni = 0; ni < N; ni++) {
            for (int ji = 0; ji < J; ji++) {
                ifs >> xTilde(ji, ni);
            }
        }
    }
    return xTilde;
}

Eigen::MatrixXd GAM::estimate(int N,
                              const NE::Model& model,
                              std::mt19937& mt) const {
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> yxs(N);
    for (int ni = 0; ni < N; ni++) {
        yxs[ni] = model.yxRvs(mt);
    }

    Eigen::MatrixXd x(model.xSize(), N);
    Eigen::MatrixXd y(model.ySize(), N);
    for (int ni = 0; ni < N; ni++) {
        x.col(ni) = yxs[ni].second;
        y.col(ni) = yxs[ni].first;
    }

    Eigen::MatrixXd xTilde = smoothing(x, y, tmpPath);
    assert(N == xTilde.cols());

    Eigen::MatrixXd fXTilde(model.fSize(), N);
    for (int ni = 0; ni < N; ni++) {
        fXTilde.col(ni) = model.f(xTilde.col(ni));
    }

    return fXTilde.rowwise().mean();
}

}  // namespace NE
