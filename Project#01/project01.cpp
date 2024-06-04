#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

const double TOLERANCE = 1e-5;

bool areEqual(double a, double b) {
    return fabs(a - b) < TOLERANCE;
}

class Molecules {
private:
    int n_atom = 0;
    vector<Vector3d> atom_pos; // Vector of Eigen::Vector3d for positions
    vector<int> charge;
    MatrixXd distance_matrix;
    vector<MatrixXd> unit_vectors;
    vector<MatrixXd> bond_angles;
    vector<vector<MatrixXd>> out_of_plane_angles;
    vector<vector<MatrixXd>> torsion_angles;
    Vector3d center_of_mass;
    Matrix3d inertia_tensor;
    Vector3d moments_of_inertia;

    MatrixXd lower_triangle_matrix(int n) {
        MatrixXd matrix = MatrixXd::Zero(n, n);
        return matrix;
    }

public:
    double getDistanceMatrix(int i, int j) {
        return distance_matrix(max(i,j), min(i,j));
    }

    Vector3d getUnitVectors(int i, int j) {
        return unit_vectors[i].col(j);
    }

    double getBondAngles(int i, int j, int k) {
        vector<int> indices = {i, j, k};
        sort(indices.begin(), indices.end());
        return bond_angles[indices[2]](indices[1], indices[0]);
    }

    double getOutOfPlaneAngles(int i, int j, int k, int l) {
        vector<int> indices = {i, j, k, l};
        sort(indices.begin(), indices.end());
        return out_of_plane_angles[indices[3]][indices[2]](indices[1], indices[0]);
    }

    double getTorsionAngles(int i, int j, int k, int l) {
        vector<int> indices = {i, j, k, l};
        sort(indices.begin(), indices.end());
        return torsion_angles[indices[3]][indices[2]](indices[1], indices[0]);
    }

    double getInertiaTensor(int i, int j) {
        return inertia_tensor(i, j);
    }

    void extractData(const filesystem::path& filepath) {
        ifstream input(filepath);
        input >> n_atom;

        atom_pos.resize(n_atom);
        charge.resize(n_atom);

        for (int i = 0; i < n_atom; ++i) {
            input >> charge[i];
            for (int j = 0; j < 3; ++j) {
                input >> atom_pos[i](j);
            }
        }
        input.close();
    }

    void calculateDistanceMatrix() {
        distance_matrix = lower_triangle_matrix(n_atom);
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                double distance = (atom_pos[i] - atom_pos[j]).norm();
                distance_matrix(i, j) = distance;
                distance_matrix(j, i) = distance;
            }
        }
    }

    void calcUnitVectors() {
        unit_vectors.resize(n_atom, MatrixXd(3, n_atom));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                Vector3d diff = atom_pos[i] - atom_pos[j];
                double length = getDistanceMatrix(i, j);
                unit_vectors[i].col(j) = diff / length;
                unit_vectors[j].col(i) = -diff / length;
            }
        }
    }

    void calculateBondAngles() {
        bond_angles.resize(n_atom, MatrixXd(n_atom, n_atom));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                for (int k = 0; k < j; ++k) {
                    bond_angles[i](j, k) = acos(unit_vectors[i].col(j).dot(unit_vectors[j].col(k)));
                }
            }
        }
    }

    void calculateOutOfPlaneAngles() {
        out_of_plane_angles.resize(n_atom, vector<MatrixXd>(n_atom, MatrixXd(n_atom, n_atom)));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < n_atom; ++j) {
                if (i == j) continue; // Skip the same atoms
                for (int k = 0; k < n_atom; ++k) {
                    if (k == i || k == j) continue; // Skip the same atoms
                    for (int l = 0; l < n_atom; ++l) {
                        if (l == i || l == j || l == k) continue; // Skip the same atoms

                        /*// Calculate the normal vector to the plane defined by j, k, and l
                        Vector3d normal = unit_vectors[j].col(k).cross(unit_vectors[j].col(l)).normalized();
                        // Calculate the out-of-plane angle
                        double angle = asin(unit_vectors[j].col(i).dot(normal) / sin(bond_angles[j](k, l)));
                        out_of_plane_angles[i][j](k, l) = angle;*/
                    }
                }
            }
        }
    }

    /*void calculateTorsionAngles() {
        torsion_angles.resize(n_atom, vector<MatrixXd>(n_atom, MatrixXd(n_atom, n_atom)));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                for (int k = 0; k < j; ++k) {
                    for (int l = 0; l < k; ++l) {
                        Vector3d cross_product = unit_vectors[i].col(j).cross(unit_vectors[j].col(k));
                        Vector3d cross_product2 = unit_vectors[j].col(k).cross(unit_vectors[k].col(l));
                        double angle1 = sin(getBondAngles(i, j, k));
                        double angle2 = sin(getBondAngles(j, k, l));
                        torsion_angles[i][j](k, l) = acos(cross_product.dot(cross_product2) / (angle1 * angle2));
                    }
                }
            }
        }
    }*/

    void calculateCenterOfMass() {
        center_of_mass.setZero();
        int total_charge = 0;
        for (int i = 0; i < n_atom; ++i) {
            center_of_mass += atom_pos[i] * charge[i];
            total_charge += charge[i];
        }
        center_of_mass /= total_charge;
    }

    void calculateInertiaTensor() {
        inertia_tensor.setZero();
        for (int i = 0; i < n_atom; ++i) {
            inertia_tensor(0, 0) += charge[i] * (atom_pos[i][0] * atom_pos[i][0]);
            inertia_tensor(1, 1) += charge[i] * (atom_pos[i][1] * atom_pos[i][1]);
            inertia_tensor(2, 2) += charge[i] * (atom_pos[i][2] * atom_pos[i][2]);
            inertia_tensor(0, 1) += charge[i] * atom_pos[i][0] * atom_pos[i][1];
            inertia_tensor(0, 2) += charge[i] * atom_pos[i][0] * atom_pos[i][2];
            inertia_tensor(1, 2) += charge[i] * atom_pos[i][1] * atom_pos[i][2];
        }
        inertia_tensor(1, 0) = inertia_tensor(0, 1);
        inertia_tensor(2, 0) = inertia_tensor(0, 2);
        inertia_tensor(2, 1) = inertia_tensor(1, 2);

        cout << "The inertia tensor is:\n" << inertia_tensor << endl;
    }
    void calculateMomentsOfInertia() {
        SelfAdjointEigenSolver<Matrix3d> es(inertia_tensor);
        moments_of_inertia = es.eigenvalues();
        cout << "The eigenvalues of the inertia tensor are:\n" << moments_of_inertia.transpose() << endl;

        if (areEqual(moments_of_inertia[0], 0) && areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            cout << (n_atom == 2 ? "The molecule is diatomic" : "The molecule is polyatomic linear.") << endl;
        } else if (!areEqual(moments_of_inertia[0], moments_of_inertia[1]) && !areEqual(moments_of_inertia[0], moments_of_inertia[2]) && !areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            cout << "The molecule is asymmetric top." << endl;
        } else if (moments_of_inertia[0] < moments_of_inertia[1] && areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            cout << "The molecule is prolate symmetric top." << endl;
        } else if (areEqual(moments_of_inertia[0], moments_of_inertia[1]) && moments_of_inertia[1] < moments_of_inertia[2]) {
            cout << "The molecule is oblate symmetric top." << endl;
        } else if (areEqual(moments_of_inertia[0], moments_of_inertia[1]) && areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            cout << "The molecule is spherical top." << endl;
        }
    }
};

int main() {
    string path = "/Users/jschmidt/Dev/ProgrammingProjects/Project#01/input";
    vector<Molecules> molecules;
    for (const auto& entry : filesystem::directory_iterator(path)) {
        if (entry.path().extension() != ".dat") {
            continue;
        }
        cout << entry.path() << endl;
        Molecules molecule;
        molecule.extractData(entry.path());
        molecule.calculateDistanceMatrix();
        molecule.calcUnitVectors();
        molecule.calculateBondAngles();
        molecule.calculateOutOfPlaneAngles();
        //molecule.calculateTorsionAngles();
        molecule.calculateCenterOfMass();
        molecule.calculateInertiaTensor();
        molecule.calculateMomentsOfInertia();
        molecules.push_back(molecule);
    }
    return 0;
}