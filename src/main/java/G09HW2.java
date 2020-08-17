import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G09HW2 {
    /**
     * Seed for generating random numbers.
     */
    private final static long SEED = 1238364L;

    /**
     * Reads a string and returns an instance of class Vector composed of double values.
     *
     * @param str is a string in the form "x, y[, z, ...]" with x, y, z real values (parsed into double values)
     * @return numerical values in the string composing the instance of class Vector
     */
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    /**
     * Given a file name, reads the file and returns a list of instances of class Vector.
     *
     * @param filename is the name of the file to read
     * @return list of instances of class Vector
     * @throws IOException if there are errors parsing the file
     */
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
            .map(str -> strToVector(str))
            .forEach(e -> result.add(e));
        return result;
    }

    /**
     * Computes the Euclidean L2-distance between two vectors represented by the class Vector.
     *
     * @param a first vector
     * @param b second vector
     * @return square root of the squared distance between a and b
     */
    public static double euclideanDistance(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    /**
     * Exact algorithm for the maximum pairwise distance problem (complexity: O(|S|^2)).
     * S is a set (implemented with an ArrayList) of points with real coordinates (i.e. Vector is composed by double values).
     *
     * @param S is a list of Vector (with double values)
     * @return exact maximum pairwise distance between all points in S
     */
    public static double exactMPD(ArrayList<Vector> S) {
        double maxDistance = 0;
        double distance;

        for (Vector p1: S) {
            for (Vector p2: S) {
                distance = euclideanDistance(p1, p2);
                if (distance > maxDistance) maxDistance = distance;
            }
        }

        return maxDistance;
    }

    /**
     * 2-approximation algorithm for the maximum pairwise distance problem (complexity: O(|S|*k)).
     * S is a set (implemented with an ArrayList) of points with real coordinates (i.e. Vector is composed by double values).
     * k is smaller than |S|.
     *
     * @param S is a list of Vector (with double values)
     * @param k integer representing the number of centers to select
     * @return 2-approximation maximum pairwise distance between all points in S
     */
    public static double twoApproxMPD(ArrayList<Vector> S, int k) {
        Random random = new Random(SEED);
        final int pointSetSize = S.size();
        ArrayList<Vector> sPrime = new ArrayList<>();
        double maxDistance = 0D;
        double distance;

        for (int i = 0; i < k; i++) {
            sPrime.add(S.get(random.nextInt(pointSetSize)));
        }


        for (Vector p1: S) {
            for (Vector p2: sPrime) {
                distance = euclideanDistance(p1, p2);
                if (distance > maxDistance) {
                    maxDistance = distance;
                }
            }
        }

        return maxDistance;
    }

    /**
     * k-center based algorithm for the maximum pairwise distance problem (complexity: O(|S|*k)).
     * S is a set (implemented with an ArrayList) of points with real coordinates (i.e. Vector is composed by double values).
     * k is smaller than |S|.
     *
     * @param S is a list of Vector (with double values)
     * @param k integer representing the number of centers to select
     * @return k-center based maximum pairwise distance between all points in S
     */
    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) {
        Random random = new Random(SEED);
        final int pointSetSize = S.size();

        // used for storing center points
        ArrayList<Vector> centers = new ArrayList<>();

        // used for storing the distances from points in S to the the last computed center
        ArrayList<Double> distances = new ArrayList<>();

        // randomly generating the first center
        centers.add(S.get(random.nextInt(pointSetSize)));

        // computing the distances between each point in p and the first computed center
        Vector firstCenter = centers.get(0);
        // complexity: O(|S|)
        for (Vector p: S) {
            distances.add(euclideanDistance(p, firstCenter));
        }

        // complexity: O(k) outer loop, O(|S|) inner loops, which results in a O(|S|*k) complexity
        for (int i = 1; i < k; i++) {
            double dMax = 0;
            int iMax = 0;
            double current;

            for (int j = 0; j < pointSetSize; j++) {
                current = distances.get(j);
                if(current != 0 && current > dMax) { // looping for each p in the set S-centers (S minus centers)
                    dMax = current;
                    iMax = j;
                }
            }

            // adding the new center
            centers.add(S.get(iMax));

            // computing the distances between each point in p and the i-th computed center
            Vector ithCenter = centers.get(i);
            for (int j = 0; j < pointSetSize; j++) {
                double currentDistance = euclideanDistance(S.get(j), ithCenter);
                if (distances.get(j) > currentDistance) {
                    distances.set(j, currentDistance);
                }
            }
        }

        return centers;
    }

    public static void main(String[] args) throws IOException {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Checking input arguments
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: file_path num_partitions");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Pointset setup
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        String filename = args[0];
        ArrayList<Vector> inputPoints = readVectorsSeq(filename);
        final int k = Integer.parseInt(args[1]);

        // k must be greater than 0 and smaller than the size of the pointset.
        if (k <= 0 || k >= inputPoints.size()) {
            throw new IllegalArgumentException("k must be greater than 0 and smaller than the size of the pointset.");
        }

        long startTime = 0L; // variable for storing starting times
        long runningTime = 0L; // variable for storing running times
        double maxDistance = 0D; // variable for storing maximum distances

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // Exact algorithm
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        startTime = System.currentTimeMillis();

        maxDistance = exactMPD(inputPoints);

        runningTime = System.currentTimeMillis() - startTime;

        System.out.println("EXACT ALGORITHM");
        System.out.println("Max distance = " + maxDistance);
        System.out.println("Running time = " + runningTime + " ms\n");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // 2-approximation algorithm
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        startTime = System.currentTimeMillis();

        maxDistance = twoApproxMPD(inputPoints, k);

        runningTime = System.currentTimeMillis() - startTime;

        System.out.println("2-APPROXIMATION ALGORITHM");
        System.out.println("k = " + k);
        System.out.println("Max distance = " + maxDistance);
        System.out.println("Running time = " + runningTime + " ms\n");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // k-center based algorithm
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        startTime = System.currentTimeMillis();

        ArrayList<Vector> centers = kCenterMPD(inputPoints, k);
        maxDistance = exactMPD(centers);

        runningTime = System.currentTimeMillis() - startTime;

        System.out.println("k-CENTER-BASED ALGORITHM");
        System.out.println("k = " + k);
        System.out.println("Max distance = " + maxDistance);
        System.out.println("Running time = " + runningTime + " ms");
    }
}
