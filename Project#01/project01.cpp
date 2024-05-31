/**
 * @file project1.cpp
 * @author Johannes Schmidt
 * @date 28.05.24
 * @brief This file contains the Molecules class and the main function for a molecular dynamics calculation.
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <Eigen/Eigenvalues>

using namespace std;


/**
 * TODO:
 * 1. Optimize saving of distance matrix, bond angles, out-of-plane angles, and torsion angles
 * 2. Check whether symmetries used in calculations are correct
 */


/**
 * @class AtomNumbers
 * @brief This class represents a collection of atoms and provides methods for calculating properties such as distance matrices, unit vectors, and bond angles.
 */

// Tolerance for comparing floating-point numbers
const double TOLERANCE = 1e-5;

bool areEqual(double a, double b) {
    return std::fabs(a - b) < TOLERANCE;
}

class Molecules {
private:
    int n_atom = 0; ///< The number of atoms
    vector< vector<double> > atom_pos; ///< 2D vector to store atom positions
    vector<int> charge; ///< Vector to store the charge of all atoms
    vector< vector<double> > distance_matrix; ///< 2D vector to store the distance matrix
    vector< vector< vector<double> > > unit_vectors; ///< 3D vector to store the unit vectors
    vector< vector< vector<double> > > bond_angles; ///< 3D vector to store the bond angles
    vector< vector< vector< vector<double> > > > out_of_plane_angles; ///< 4D vector to store the out-of-plane angles
    vector< vector< vector< vector<double> > > > torsion_angles; ///< 4D vector to store the torsion angles
    vector<float> center_of_mass;
    vector< vector<float> > inertia_tensor;
    vector<float> moments_of_inertia;

    vector< vector<double> > lower_triangle_matrix(int n) {
        vector< vector<double> > matrix(n, vector<double>(n));
        for (int i = 0; i < n; ++i) {
            matrix[i].resize(i+1);
        }
        return matrix;
    }
public:
    double getDistanceMatrix(int i, int j) {
        return distance_matrix[max(i,j)][min(i,j)];
    }
    double getUnitVectors(int i, int j, int k) {
        return unit_vectors[i][j][k];
    }
    double getBondAngles(int i, int j, int k) {
        vector<int> indices = {i, j, k};
        sort(indices.begin(), indices.end());

        return bond_angles[indices[2]][indices[1]][indices[0]];
    }
    double getOutOfPlaneAngles(int i, int j, int k, int l) {
        vector<int> indices = {i, j, k, l};
        // Sort the data vector
        sort(indices.begin(), indices.end());

        return out_of_plane_angles[indices[3]][indices[2]][indices[1]][indices[0]];
    }
    double getTorsionAngles(int i, int j, int k, int l) {
        vector<int> indices = {i, j, k, l};
        // Sort the data vector
        sort(indices.begin(), indices.end());

        return torsion_angles[indices[3]][indices[2]][indices[1]][indices[0]];
    }
    double getInertiaTensor(int i, int j) {
        if (i < j) {
            return inertia_tensor[j][i];
        }
        return inertia_tensor[i][j];
    }

    /**
     * @brief Extracts atom data from a file.
     * @param filename The name of the file to extract data from.
     */
    void extractData(string filename) {
        ifstream input(filename);
        input >> n_atom;

        atom_pos.resize(n_atom, vector<double>(3, 0.0));// Resize the 2D vector to hold the positions of all atoms
        charge.resize(n_atom); // Resize the vector to hold the charge of all atoms

        for (int i = 0; i < n_atom; ++i) {
            input >> charge[i];
            for (int j = 0; j < 3; ++j) { // Loop to read the 3 position values for each atom
                input >> atom_pos[i][j];
            }
        }
        input.close();
    }

    /**
     * @brief Calculates the distance matrix for the atoms.
     */
    void calculateDistanceMatrix() {
        distance_matrix = lower_triangle_matrix(n_atom);
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                double distance = 0;
                for (int k = 0; k < 3; ++k) {
                    distance += (atom_pos[i][k] - atom_pos[j][k]) * (atom_pos[i][k] - atom_pos[j][k]);
                }
                distance_matrix[i][j] = sqrt(distance);
            }
        }
    }

    /**
     * @brief Calculates the unit vectors for each pair of atoms.
     */
    void calcUnitVectors() {
        unit_vectors.resize(n_atom, vector<vector<double> >(n_atom, vector<double>(3, 0.0)));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                vector<double> diff = {atom_pos[i][0] - atom_pos[j][0], atom_pos[i][1] - atom_pos[j][1], atom_pos[i][2] - atom_pos[j][2]};
                double length = getDistanceMatrix(i, j);
                unit_vectors[i][j] = {diff[0] / length, diff[1] / length, diff[2] / length};
                unit_vectors[j][i] = {-diff[0] / length, -diff[1] / length, -diff[2] / length};
            }
        }
    }

    /**
     * @brief Calculates the bond angles for each triplet of atoms.
     */
    void calculateBondAngles() {
        bond_angles.resize(n_atom, vector<vector<double> >(n_atom, vector<double>(n_atom, 0.0)));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                for (int k = 0; k < j; ++k) {
                    bond_angles[i][j][k] = acos(unit_vectors[i][j][0]*unit_vectors[j][k][0] + unit_vectors[i][j][1]*unit_vectors[j][k][1] + unit_vectors[i][j][2]*unit_vectors[j][k][2]);
                }
            }
        }
    }
    void calculateOutOfPlaneAngles() {
        out_of_plane_angles.resize(n_atom, vector< vector<vector<double> > >(n_atom, vector< vector<double> >(n_atom, vector<double>(n_atom, 0.0))));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                for (int k = 0; k < j; ++k) {
                    for (int l = 0; l < k; ++l) {
                        vector<double> cross_product = {unit_vectors[k][j][1]*unit_vectors[k][l][2] - unit_vectors[k][j][2]*unit_vectors[k][l][1],
                                                        unit_vectors[k][j][2]*unit_vectors[k][l][0] - unit_vectors[k][j][0]*unit_vectors[k][l][2],
                                                        unit_vectors[k][j][0]*unit_vectors[k][l][1] - unit_vectors[k][j][1]*unit_vectors[k][l][0]};
                        out_of_plane_angles[i][j][k][l] = acos((cross_product[0]*unit_vectors[k][i][0] + cross_product[1]*unit_vectors[k][i][1] + cross_product[2]*unit_vectors[k][i][2])/sin(bond_angles[j][k][l]));
                    }
                }
            }
        }
    }
    void calculateTorsionAngles() {
        torsion_angles.resize(n_atom, vector< vector<vector<double> > >(n_atom, vector< vector<double> >(n_atom, vector<double>(n_atom, 0.0))));
        for (int i = 0; i < n_atom; ++i) {
            for (int j = 0; j < i; ++j) {
                for (int k = 0; k < j; ++k) {
                    for (int l = 0; l < k; ++l) {
                        vector<double> cross_product = {unit_vectors[i][j][1] * unit_vectors[j][k][2] -
                                                        unit_vectors[i][j][2] * unit_vectors[j][k][1],
                                                        unit_vectors[i][j][2] * unit_vectors[j][k][0] -
                                                        unit_vectors[i][j][0] * unit_vectors[j][k][2],
                                                        unit_vectors[i][j][0] * unit_vectors[j][k][1] -
                                                        unit_vectors[i][j][1] * unit_vectors[j][k][0]};
                        vector<double> cross_product2 = {unit_vectors[j][k][1] * unit_vectors[k][l][2] -
                                                         unit_vectors[j][k][2] * unit_vectors[k][l][1],
                                                         unit_vectors[j][k][2] * unit_vectors[k][l][0] -
                                                         unit_vectors[j][k][0] * unit_vectors[k][l][2],
                                                         unit_vectors[j][k][0] * unit_vectors[k][l][1] -
                                                         unit_vectors[j][k][1] * unit_vectors[k][l][0]};
                        double angle1 = sin(getBondAngles(i, j, k));
                        double angle2 =sin(getBondAngles(j, k, l));
                        torsion_angles[i][j][k][l] = acos((cross_product[0] * cross_product2[0] +
                                                           cross_product[1] * cross_product2[1] +
                                                           cross_product[2] * cross_product2[2]) /
                                                          (angle1 * angle2));
                    }
                }
            }
        }
    }
    void calculateCenterOfMass() {
        center_of_mass.resize(3, 0.0);
        int weight = 0;
        for (int i = 0; i < n_atom; i++) {
            for (int j = 0; j < 3; j++) {
                weight += charge[i];
                center_of_mass[j] = atom_pos[i][j] * charge[i];
            }
        }
    }
    void calculateInertiaTensor() {
        inertia_tensor.resize(3, vector<float>(3, 0.0));

        for (int i = 0; i < n_atom; i++) {
            inertia_tensor[0][0] += charge[i] * (atom_pos[i][1] * atom_pos[i][1] + atom_pos[i][2] * atom_pos[i][2]);
            inertia_tensor[1][1] += charge[i] * (atom_pos[i][0] * atom_pos[i][0] + atom_pos[i][2] * atom_pos[i][2]);
            inertia_tensor[2][2] += charge[i] * (atom_pos[i][0] * atom_pos[i][0] + atom_pos[i][1] * atom_pos[i][1]);
            inertia_tensor[0][1] += charge[i] * atom_pos[i][0] * atom_pos[i][1];
            inertia_tensor[0][2] += charge[i] * atom_pos[i][0] * atom_pos[i][2];
            inertia_tensor[1][2] += charge[i] * atom_pos[i][1] * atom_pos[i][2];
        }
        cout << "The inertia tensor is:\n";
        cout << inertia_tensor[0][0] << " " << inertia_tensor[0][1] << " " << inertia_tensor[0][2] << endl;
        cout << inertia_tensor[0][1] << " " << inertia_tensor[1][1] << " " << inertia_tensor[1][2] << endl;
        cout << inertia_tensor[0][2] << " " << inertia_tensor[1][2] << " " << inertia_tensor[2][2] << endl;
    }
    void calculateMomentsOfInertia() {
        moments_of_inertia.resize(3);
        Eigen::MatrixXd A(3, 3);
        A << inertia_tensor[0][0], inertia_tensor[0][1], inertia_tensor[0][2], // Initialize the matrix A with the values of the inertia tensor, symmetric matrix
                inertia_tensor[0][1], inertia_tensor[1][1], inertia_tensor[1][2],
                inertia_tensor[0][2], inertia_tensor[1][2], inertia_tensor[2][2];

        Eigen::EigenSolver<Eigen::MatrixXd> es(A);
        std::cout << "The eigenvalues of A are:\n" << es.eigenvalues().real() << std::endl;
        moments_of_inertia[0] = es.eigenvalues()[0].real();
        moments_of_inertia[1] = es.eigenvalues()[1].real();
        moments_of_inertia[2] = es.eigenvalues()[2].real();
        if (areEqual(moments_of_inertia[0], 0) && areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            if (n_atom == 2) {
                std::cout << "The molecule is diatomic" << std::endl;
            } else {
                std::cout << "The molecule is polyatomic linear" << std::endl;
            }
        } else if (!areEqual(moments_of_inertia[0], moments_of_inertia[1]) && !areEqual(moments_of_inertia[0], moments_of_inertia[2]) && !areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            std::cout << "The molecule is asymetric top" << std::endl;
        } else if (moments_of_inertia[0] < moments_of_inertia[1] && areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            std::cout << "The molecule is prolate symmetric top" << std::endl;
        } else if (areEqual(moments_of_inertia[0], moments_of_inertia[1]) && moments_of_inertia[1] < moments_of_inertia[2]) {
            std::cout << "The molecule is oblate symmetric top" << std::endl;
        } else if (areEqual(moments_of_inertia[0], moments_of_inertia[1]) && areEqual(moments_of_inertia[1], moments_of_inertia[2])) {
            std::cout << "The molecule is spherical top" << std::endl;
        }
    }
};

/**
 * @brief The main function of the program. It reads atom data from files, calculates properties, and stores the results in AtomNumbers objects.
 * @return 0 if the program runs successfully, and non-zero otherwise.
 */
int main() {
    std::string path = "/Users/jschmidt/Dev/ProgrammingProjects/Project#01/input";
    std::vector<Molecules> molecules; // Vector to store AtomNumbers objects

    for (const auto & entry : std::filesystem::directory_iterator(path)) {
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
        molecule.calculateTorsionAngles();
        molecule.calculateCenterOfMass();
        molecule.calculateInertiaTensor();
        molecule.calculateMomentsOfInertia();
        molecules.push_back(molecule); // Add the AtomNumbers object to the vector
    }
    return 0;
}