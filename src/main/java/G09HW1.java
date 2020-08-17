import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.Serializable;

import java.io.IOException;
import java.util.*;

/**
 * Comparator class for Tuple2<String, Long> objects.
 * The comparation is done at key level.
 */
class LongComparator implements Serializable, Comparator<Tuple2<String, Long>> {
  private static final long serialVersionUID = 1L; // implementing Serializable requires adding a serialVersionUID

  @Override
  public int compare(Tuple2<String, Long> o1, Tuple2<String, Long> o2) {
    return Long.compare(o1._2(), o2._2());
  }
}

public class G09HW1 {

  public static void main(String[] args) throws IOException {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // CHECKING NUMBER OF CMD LINE PARAMETERS
    // Parameters are: number_partitions, <path to file>
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    if (args.length != 2) {
      throw new IllegalArgumentException("USAGE: num_partitions file_path");
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // SPARK SETUP
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    SparkConf conf = new SparkConf(true).setAppName("Homework1");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("WARN");

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // INPUT READING
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    // Read number of partitions
    int K = Integer.parseInt(args[0]);

    // Read input file and subdivide it into K random partitions
    JavaRDD<String> pairStrings = sc.textFile(args[1]).repartition(K);

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // SETTING GLOBAL VARIABLES
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    JavaPairRDD<String, Long> classCount;

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // INPUT
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    System.out.println("INPUT:");
    System.out.println("\n** K=" + K);
    System.out.println("\n** DATASET: " + args[1]);

    System.out.println("\nOUTPUT:");

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Class count with deterministic partitions
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    System.out.println("\nVERSION WITH DETERMINISTIC PARTITIONS");

    classCount = pairStrings.mapToPair((pair) -> { // <-- MAP PHASE (R1)
      /*
       * The input pair is a string: pair = i + " " + gamma_i
       */
      String[] a = pair.split(" ");
      Long key = Long.parseLong(a[0]) % K;
      String value = a[1];

      return new Tuple2<>(key, value);
    }).groupByKey() // <-- REDUCE PHASE (R1)
        .flatMapToPair((pair) -> {
          /*
           * For each j in [0, K): the map counts will contain pairs (gamma, c_j(gamma))
           * where c_j(gamma) is the number of objects of class gamma in the subset S_j.
           *
           * counts.get(gamma) corresponds to c_j(gamma)
           */
          HashMap<String, Long> counts = new HashMap<>();
          for (String gamma : pair._2()) {
            counts.put(gamma, 1L + counts.getOrDefault(gamma, 0L));
          }

          // Creating the output pairs for R1
          ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
          for (Map.Entry<String, Long> e : counts.entrySet()) {
            pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
          }

          return pairs.iterator();
        }).groupByKey() // <-- REDUCE PHASE (R2)
        .mapValues((it) -> {
          long sum = 0L;

          for (long c : it) {
            sum += c;
          }

          return sum;
        });

    /*
     * Printing the output. Note: sortByKey(true, 1) tells the RDD to group all
     * pairs in 1 partition and sort them alphabetically (in this particular case)
     */

    System.out.print("Output pairs = ");
    classCount.sortByKey(true, 1).foreach(pair -> System.out.print(pair + " "));

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // VERSION WITH SPARK PARTITIONS
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    System.out.println("\nVERSION WITH SPARK PARTITIONS");
    classCount = pairStrings.flatMapToPair((pair) -> { // <-- MAP PHASE (R1)
      // The input pair is a string: pair = i + " " + gamma_i
      String[] a = pair.split(" ");
      Long key = Long.parseLong(a[0]) % K;
      String value = a[1];

      ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>();
      pairs.add(new Tuple2<>(key, value));

      return pairs.iterator();
    }).mapPartitionsToPair((partition) -> { // <-- REDUCE PHASE (R1)
      /*
       * For each j in [0, K): the map counts will contain pairs (gamma, c_j(gamma))
       * where c_j(gamma) is the number of objects of class gamma in the subset S_j.
       *
       * counts.get(gamma) corresponds to c_j(gamma)
       */
      HashMap<String, Long> counts = new HashMap<>();

      while (partition.hasNext()) {
        String gamma = partition.next()._2();
        counts.put(gamma, 1L + counts.getOrDefault(gamma, 0L));
      }

      // Creating the output pairs for R1
      ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

      // Counter for the number of pairs in the current partition
      Long numberOfPairs = 0L;

      for (Map.Entry<String, Long> e : counts.entrySet()) {
        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
        numberOfPairs += e.getValue();
      }

      // Creating the extra pair for the partitionSize
      pairs.add(new Tuple2<>("maxPartitionSize", numberOfPairs));

      return pairs.iterator();
    }).groupByKey() // <-- REDUCE PHASE (R2)
        .flatMapToPair((pair) -> {
          // pairs is the set of pairs after Round 1
          ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
          
          /*
           * Two possibile pairs exist:
           * 1. (maxPartitionSize, partitionSize)
           * 2. (gamma, c_j(gamma))
           * 
           * Case 1: the pair with the highest value must be returned
           * Case 2: for each gamma, a pair (gamma, sum_{j=0}^{K-1} c_j(gamma)) must be returned
           */
          if (pair._1().equals("maxPartitionSize")) {
            Long maxSize = 0L;

            for (Long currentPartitionSize : pair._2()) {
              if (currentPartitionSize > maxSize) {
                maxSize = currentPartitionSize;
              }
            }

            pairs.add(new Tuple2<>("maxPartitionSize", maxSize));
          } else {
            Long sum = 0L;

            for (Long c : pair._2()) {
              sum += c;
            }

            pairs.add(new Tuple2<>(pair._1(), sum));
          }

          return pairs.iterator();
        });
    
    /*
     * Gathering the most frequent class:
     * 1. if the most frequent class is the one with "fake class" maxPartitionSize of course we do not want to print it, thus we'll exclude it
     * 2. if there are classes with equivalent sizes then we want the first one in alphabetical order by class, thus we'll order the output set
     * 3. we get the pair with the maximum value
     */
    Tuple2<String, Long> mostFrequentClass = classCount.filter(pair -> !pair._1().equals("maxPartitionSize")).sortByKey(true, 1)
        .max(new LongComparator());

    // Gathering the maxPartitionSize
    Long maxPartitionSize = classCount.filter(t -> t._1().equals("maxPartitionSize")).first()._2();

    System.out.println("Most frequent class = " + mostFrequentClass);
    System.out.println("Max partition size = " + maxPartitionSize);
  }
}
