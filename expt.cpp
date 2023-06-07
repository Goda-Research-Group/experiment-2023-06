#include <random>

#include "all.cpp"

std::vector<int> Ns = {1000, 10000, 100000, 1000000};
std::vector<int> Nsb = {6, 9, 13, 17, 20};
int R = 100;

int main() {
    std::mt19937 mt(1234);
    int c;

    EVSI::TestCase m11(c, 7, 0.7);
    EVSI::TestCase m12(c, 7, 0.9);
    EVSI::TestCase m13(c, 10, 0.7);
    EVSI::TestCase m14(c, 10, 0.9);
    EVSI::Bluebelle m21(c, 100, 100, 50, 0.001);
    EVSI::Bluebelle m22(c, 200, 200, 100, 0.001);
    EVSI::Bluebelle m23(c, 400, 400, 200, 0.001);
    EVSI::Bluebelle m24(c, 800, 800, 400, 0.001);
    EVSI::Bluebelle m31(c, 50, 50, 50, 50);
    EVSI::Bluebelle m32(c, 125, 125, 125, 125);
    EVSI::Bluebelle m33(c, 250, 250, 250, 250);
    EVSI::Bluebelle m34(c, 500, 500, 500, 500);

    NE::ProposedSimple e1;
    NE::ProposedSparseGrid e2;
    NE::GAM e3("/dev/shm");

    visualize(Nsb, R, e1, m11, c, mt);
    visualize(Nsb, R, e2, m11, c, mt);
    visualize(Ns, R, e3, m11, c, mt);
    visualize(Nsb, R, e1, m12, c, mt);
    visualize(Nsb, R, e2, m12, c, mt);
    visualize(Ns, R, e3, m12, c, mt);
    visualize(Nsb, R, e1, m13, c, mt);
    visualize(Nsb, R, e2, m13, c, mt);
    visualize(Ns, R, e3, m13, c, mt);
    visualize(Nsb, R, e1, m14, c, mt);
    visualize(Nsb, R, e2, m14, c, mt);
    visualize(Ns, R, e3, m14, c, mt);

    visualize(Nsb, R, e1, m21, c, mt);
    visualize(Nsb, R, e2, m21, c, mt);
    visualize(Ns, R, e3, m21, c, mt);
    visualize(Nsb, R, e1, m22, c, mt);
    visualize(Nsb, R, e2, m22, c, mt);
    visualize(Ns, R, e3, m22, c, mt);
    visualize(Nsb, R, e1, m23, c, mt);
    visualize(Nsb, R, e2, m23, c, mt);
    visualize(Ns, R, e3, m23, c, mt);
    visualize(Nsb, R, e1, m24, c, mt);
    visualize(Nsb, R, e2, m24, c, mt);
    visualize(Ns, R, e3, m24, c, mt);

    visualize(Nsb, R, e1, m31, c, mt);
    visualize(Nsb, R, e2, m31, c, mt);
    visualize(Ns, R, e3, m31, c, mt);
    visualize(Nsb, R, e1, m32, c, mt);
    visualize(Nsb, R, e2, m32, c, mt);
    visualize(Ns, R, e3, m32, c, mt);
    visualize(Nsb, R, e1, m33, c, mt);
    visualize(Nsb, R, e2, m33, c, mt);
    visualize(Ns, R, e3, m33, c, mt);
    visualize(Nsb, R, e1, m34, c, mt);
    visualize(Nsb, R, e2, m34, c, mt);
    visualize(Ns, R, e3, m34, c, mt);

    visualize(Nsb, R, e1, m11, c, mt);
    visualize(Nsb, R, e2, m11, c, mt);
    visualize(Ns, R, e3, m11, c, mt);
    visualize(Nsb, R, e1, m12, c, mt);
    visualize(Nsb, R, e2, m12, c, mt);
    visualize(Ns, R, e3, m12, c, mt);
    visualize(Nsb, R, e1, m13, c, mt);
    visualize(Nsb, R, e2, m13, c, mt);
    visualize(Ns, R, e3, m13, c, mt);
    visualize(Nsb, R, e1, m14, c, mt);
    visualize(Nsb, R, e2, m14, c, mt);
    visualize(Ns, R, e3, m14, c, mt);
}
