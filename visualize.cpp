
void visualize(std::vector<int> Ns,
               int R,
               const NE::Estimator& e,
               const NE::Model& m,
               int& counter,
               std::mt19937& mt) {
    std::ostringstream outputFile;
    outputFile << "output/" << m.getName() << "_" << e.getName() << ".txt";
    FILE* fp1 = fopen(outputFile.str().c_str(), "w");
    if (fp1 == NULL) {
        throw std::runtime_error("Error: Cannot open the file.\n");
    }

    for (int n : Ns) {
        Eigen::MatrixXd errors(m.getTheoretical().rows(), R);
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> num(1, R);
        Eigen::Matrix theoretical = m.getTheoretical();
        for (int ri = 0; ri < R; ri++) {
            counter = 0;
            auto result = e.estimate(n, m, mt);
            num(0, ri) = counter;
            errors.col(ri) = result - theoretical;
        }
        double mse = (errors.colwise().squaredNorm()).mean();
        double smean = (errors.rowwise().mean()).squaredNorm();
        double var = mse - smean;
        int nmean = num.mean();
        fprintf(fp1, "%10d %+.12f %+.12f %+.12f\n", nmean, mse, smean, var);
    }

    fclose(fp1);
}
