import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import org.apache.spark.api.java.JavaRDD;

import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

public class G09HW3 {
    /**
     * Seed for generating random numbers.
     */
    private final static long SEED = 1238364L;

    /**
     * Duration (in ms) of the execution of Round 1 of runMapReduce
     */
    private static Long executionRoundOne;

    /**
     * Duration (in ms) of the execution of Round 2 of runMapReduce
     */
    private static Long executionRoundTwo;

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
                if (current != 0 && current > dMax) { // looping for each p in the set S-centers (S minus centers)
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
                if(distances.get(j) > currentDistance) {
                    distances.set(j, currentDistance);
                }
            }
        }

        return centers;
    }

    /**
     * Diversity Maximization algorithm - Sequential 2-approximation based on matching
     * points is a set (implemented with an ArrayList) of points with real coordinates (i.e. Vector is composed by double values).
     * k is smaller than |points|.
     *
     * @param points is a list of Vector (with double values)
     * @param k integer representing the number of points to maximize their average distance
     * @return list of k points which have the maximum average distance
     */
    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {
        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);

        for (int iter = 0; iter < k / 2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i + 1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }

        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }

        return result;
    }

    /**
     * Diversity Maximization algorithm - Coreset-based 4-approximation MapReduce algorithm
     * pointsRDD is a distributed dataset (implemented with aa JavaRDD) of points with real coordinates (i.e. Vector is composed by double values).
     * k is smaller than |points|.
     * L is the number of partition in which k points get extracted.
     *
     * @param pointsRDD is a set of Vector (with double values)
     * @param k number of points to select in each partition
     * @param L number of partitions
     * @return solution of the algorithm
     */
    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L) {
        // -----------
        // | Round 1 |
        // -----------

        // Goal of Round 1: Subdividing pointsRDD into L partitions and extracting k points from each partition
        // using the Farthest-First Traversal algorithm.
        // In this case we assume the partitioning has already been made, otherwise pointsRDD.repartition(L) would need to be invoked.

        long startTime = System.currentTimeMillis();

        JavaRDD<Vector> coresetRDD = pointsRDD.mapPartitions( // mapPartitions is a lazy operation
                (partition) -> {
                    ArrayList<Vector> part = new ArrayList<>();
                    partition.forEachRemaining(part::add);
                    ArrayList<Vector> centers = kCenterMPD(part, k);
                    return centers.iterator();
                }
        ).persist(StorageLevel.MEMORY_AND_DISK());
        // persist(StorageLevel.MEMORY_AND_DISK()) works in the same way
        // as persist(StorageLevel.MEMORY_ONLY()) and cache() if no I/O operations to disk are involved.

        long coresetSize = coresetRDD.count(); // count is an action (not lazy), therefore the program needs to execute mapPartitions

        executionRoundOne = System.currentTimeMillis() - startTime;

        // -----------
        // | Round 2 |
        // -----------

        // Goal of Round 2: Collecting the k points extracted in Round 1 from each one of the L partitions
        // (which form the coreset) into a set (implemented with an ArrayList) called coreset and returning,
        // as output, the k points computed by runSequential(coreset, k) which solve the problem.

        startTime = System.currentTimeMillis();

        ArrayList<Vector> coreset = new ArrayList<>(coresetRDD.collect());

        ArrayList<Vector> result = runSequential(coreset, k);

        executionRoundTwo = System.currentTimeMillis() - startTime;

        return result;
    }

    /**
     * Measures the exact average distance between the points belonging to the pointset given as input.
     *
     * @param pointsSet set of points in the euclidean space
     * @return average distance between all pairs of points
     */
    public static double measure(ArrayList<Vector> pointsSet) {
        final int pointSetSize = pointsSet.size();
        final double numberPairwiseDistances = (pointSetSize * (pointSetSize - 1) * 0.5);
        double totalSum = 0;

        for (int i = 0; i < pointSetSize; i++) {
            Vector ithPoint = pointsSet.get(i);
            for (int j = i + 1; j < pointSetSize; j++) {
                totalSum += euclideanDistance(ithPoint, pointsSet.get(j));
            }
        }

        return totalSum / numberPairwiseDistances;
    }

    public static void main(String[] args) {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CHECKING NUMBER OF CMD LINE PARAMETERS
        // Parameters are:  path_to_file parameter_for_diversity_maximization number_of_partitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: path_to_file parameter_for_diversity_maximization number_of_partitions");
        }

        String pathToFile = args[0];
        int k = Integer.parseInt(args[1]);
        int L = Integer.parseInt(args[2]);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        long startTime = System.currentTimeMillis();

        JavaRDD<Vector> inputPoints = sc.textFile(pathToFile).map(G09HW3::strToVector).repartition(L).cache();

        long inputSize = inputPoints.count();

        long stopTime = System.currentTimeMillis();

        System.out.println("Number of points = " + inputSize);
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (stopTime - startTime) + " ms");

        ArrayList<Vector> solution = runMapReduce(inputPoints, k, L);

        System.out.println("Runtime of Round 1 = " + executionRoundOne + " ms");
        System.out.println("Runtime of Round 2 = " + executionRoundTwo + " ms");

        double avgDistance = measure(solution);

        System.out.println("Average distance = " + avgDistance);
    }
}
