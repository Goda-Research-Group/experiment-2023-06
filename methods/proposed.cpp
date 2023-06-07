
int pow2(int x) {
    int r = 1;
    for (int xi = 0; xi < x; xi++) {
        r <<= 1;
    }
    return r;
}

static Eigen::MatrixXd meanVector(const std::vector<Eigen::MatrixXd>& A) {
    int N = A.size();
    Eigen::MatrixXd result(1, 1);
    result(0, 0) = 0;
    for (int ni = 0; ni < N; ni++) {
        result(0, 0) += A[ni](0, 0);
    }
    return result.array() /= N;
}

static std::vector<Eigen::MatrixXd> fD(const NE::Model& model,
                                       const std::vector<Eigen::MatrixXd>& A) {
    int N = A.size();
    std::vector<Eigen::MatrixXd> result(N);
    for (int ni = 0; ni < N; ni++) {
        result[ni] = model.f(A[ni]);
    }
    return result;
}

static std::vector<std::vector<Eigen::MatrixXd>>
constructBinTrees(const NE::Model& model, int N, std::mt19937& mt) {
    std::vector<Eigen::MatrixXd> x(N), y(N);
    for (int ni = 0; ni < N; ni++) {
        auto yx = model.yxRvs(mt);
        x[ni] = yx.second;
        y[ni] = yx.first;
    }
    int J = model.xSize();
    int K = model.ySize();
    std::vector<std::vector<Eigen::MatrixXd>> result(J);
    for (int ji = 0; ji < J; ji++) {
        result[ji] = std::vector<Eigen::MatrixXd>(N);
        for (int ni = 0; ni < N; ni++) {
            result[ji][ni] = Eigen::MatrixXd(3 + K, 1);
            result[ji][ni](0, 0) = x[ni](ji, 0);
            result[ji][ni](1, 0) = ni;
            result[ji][ni].bottomRows(K) = y[ni];
        }
        for (int depth = 0, sizeChunk = N; sizeChunk != 0;
             sizeChunk >>= 1, depth++) {
            for (int noi = 0; noi < N / sizeChunk; noi++) {
                sort(result[ji].begin() + sizeChunk * noi,
                     result[ji].begin() + sizeChunk * noi + sizeChunk,
                     [&](auto const& lhs, auto const& rhs) {
                         return lhs(3 + (depth % K), 0) <
                                rhs(3 + (depth % K), 0);
                     });
            }
        }
    }
    return result;
}

static void destructBinTrees(std::vector<std::vector<Eigen::MatrixXd>>& bts,
                             int sizeChunk) {
    int D = bts.size();
    int N = bts[0].size();
    for (int di = 0; di < D; di++) {
        for (int noi = 0; noi < N / sizeChunk; noi++) {
            int sizeChunkSub = sizeChunk >> 1;
            std::vector<Eigen::MatrixXd> left(sizeChunkSub),
                right(sizeChunkSub);
            for (int ci = 0; ci < sizeChunkSub; ci++) {
                left[ci] = bts[di][sizeChunk * noi + ci];
                right[ci] = bts[di][sizeChunk * noi + sizeChunkSub + ci];
            }
            auto leftItr = left.begin(), rightItr = right.begin();
            for (int ci = 0; ci < sizeChunk; ci++) {
                bool isLeft;
                if (leftItr == left.end()) {
                    isLeft = false;
                } else if (rightItr == right.end()) {
                    isLeft = true;
                } else if ((*leftItr)(1, 0) > (*rightItr)(1, 0)) {
                    isLeft = false;
                } else {
                    isLeft = true;
                }
                if (isLeft == true) {
                    bts[di][sizeChunk * noi + ci] = *leftItr;
                    leftItr++;
                } else {
                    bts[di][sizeChunk * noi + ci] = *rightItr;
                    rightItr++;
                }
            }
        }
    }
}

static std::vector<Eigen::MatrixXd> smoothingBinTree(
    const std::vector<Eigen::MatrixXd>& bt,
    int sizeChunk) {
    int N = bt.size();
    std::vector<Eigen::MatrixXd> result(N, Eigen::MatrixXd(2, 1));
    for (int noi = 0; noi < N / sizeChunk; noi++) {
        double ave = 0;
        for (int nii = 0; nii < sizeChunk; nii++) {
            ave += bt[noi * sizeChunk + nii](0, 0);
        }
        ave /= sizeChunk;
        for (int nii = 0; nii < sizeChunk; nii++) {
            result[noi * sizeChunk + nii](0, 0) = ave;
            result[noi * sizeChunk + nii](1, 0) =
                bt[noi * sizeChunk + nii](1, 0);
        }
    }
    return result;
}

static Eigen::MatrixXd estimateChunks(
    const NE::Model& model,
    std::vector<std::vector<Eigen::MatrixXd>>& bts,
    int sizeChunk) {
    int D = bts.size();
    int N = bts[0].size();
    std::vector<Eigen::MatrixXd> xTilde(N, Eigen::MatrixXd(D, 1));
    for (int di = 0; di < D; di++) {
        auto xeTilde = smoothingBinTree(bts[di], sizeChunk);
        sort(xeTilde.begin(), xeTilde.end(),
             [&](auto const& lhs, auto const& rhs) {
                 return lhs(1, 0) < rhs(1, 0);
             });
        for (int ni = 0; ni < N; ni++) {
            xTilde[ni](di, 0) = xeTilde[ni](0, 0);
        }
    }
    return meanVector(fD(model, xTilde));
}

Eigen::MatrixXd NE::ProposedSimple::estimate(int m,
                                             const NE::Model& model,
                                             std::mt19937& mt) const {
    assert(model.fSize() == 1);
    auto bts = constructBinTrees(model, pow2(m), mt);
    return estimateChunks(model, bts, pow2(m / 2));
}

Eigen::MatrixXd NE::ProposedSparseGrid::estimate(int m,
                                                 const NE::Model& model,
                                                 std::mt19937& mt) const {
    assert(model.fSize() == 1);
    auto N = pow2(m);
    auto bts = constructBinTrees(model, pow2(m), mt);
    Eigen::MatrixXd ft = estimateChunks(model, bts, 1);
    for (int sizeChunk = 1; sizeChunk < N; sizeChunk <<= 1) {
        destructBinTrees(bts, sizeChunk << 1);
        ft -= estimateChunks(model, bts, sizeChunk);
        ft += estimateChunks(model, bts, sizeChunk << 1);
    }
    return ft;
}
