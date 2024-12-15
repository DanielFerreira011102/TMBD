# Exame Solutions

## (3.0) 1. Answer the following questions: 

### (a) Explain how relevant is a family of hash functions to a Bloom filter algorithm. 

#### Confidence: 95%

A family of hash functions is essential to a Bloom filter algorithm because it determines how elements are mapped into the bit array. When we insert an element, we use multiple different hash functions to set multiple bits to 1. When we query an element, we check if all the bits set by the hash functions are 1. If any of them are 0, we know the element is not in the filter with certainty (no false negatives). If all of them are 1, the element might be in the filter but we can't be sure (possible false positives). 

False positives are possible because multiple elements can hash to the same bits, but false negatives are not possible because if an element is in the filter, all its bits will be set to 1.

Using too few hash functions means we don't set enough bits, making it easier for different elements to appear the same. On the other hand, using too many hash functions means we set too many bits to 1, eventually causing most queries to return false positives as the bit array becomes saturated. The optimal number of hash functions (to minimize false positives) depends on the size of the bit array ($m$) and the expected number of elements ($n$), calculated as: 

$$k = \frac{m}{n} \ln(2)$$

A good family of hash functions should satisfy the following conditions:

- **Uniformity**: The hash functions should distribute values uniformly across the bit array to reduce collisions.

- **Independence**: The hash functions should behave independently to avoid biased mappings.

- **Efficiency**: Computing the hash functions should be computationally inexpensive since multiple hash values are computed per operation.

For instance, cryptographic hash functions like MD5 or SHA-256 can provide high-quality hashes but are often overkill in terms of computational cost. Non-cryptographic options like MurmurHash or CityHash are typically better suited for Bloom filters.

### (b) State the CAP theorem and its consequences on storing/accessing geographic data. 

#### Confidence: 85%

The CAP theorem states that in a distributed system, it's impossible to simultaneously guarantee all three of these properties: 

- **Consistency (C)**: all nodes see the same data at the same time;

- **Availability (A)**: every request gets a response  (it may not be the most up-to-date);

- **Partition tolerance (P)**: the system continues to work even if network communication fails between nodes.

For geographic data, this means we often have to choose between:

- Having consistent geographic data across all servers but potentially being unavailable during network issues;

- Having high availability but possibly showing different geographic data on different servers;

- Sacrificing some consistency or availability to handle network partitions and ensure the system remains operational.

Due to the nature of distributed systems, partition tolerance is often mandatory (network failures are inevitable). Most geographic systems prioritize availability and partition tolerance over strict consistency, accepting that location data might be slightly outdated or inconsistent across different servers.


### (c) Identify the different perspectives of Big Data, in particular, explain in detail the perspective *HighVelocity*.

#### Confidence: 95%

Big Data is often broken down into 4 main dimensions or perspectives, commonly referred to as the 4 V's:

- **Volume**: the amount of data being generated and stored;

- **Velocity**: the speed at which data is generated, transmitted, and processed;

- **Variety**: the different types of data being collected (structured, unstructured, semi-structured);

- **Veracity**: the quality and trustworthiness of the data.

Some frameworks also include **Value** to represent the usefulness of the data in decision-making.

High Velocity refers to the speed at which data is being generated and the rate at which it must be processed. This perspective is often associated with real-time data streams, IoT devices, social media feeds, financial transactions, etc.

For example, a traffic monitoring system needs to process thousands of sensor readings per second to provide real-time traffic updates. The challenge isn't just storing this data, but processing it fast enough to make immediate decisions. This often requires specialized streaming algorithms and technologies like Apache Kafka, Spark Streaming, or Storm that can handle high-throughput data streams.

By contrast, the volume perspective focuses on handling petabytes or exabytes of data stored across distributed systems, while the variety perspective deals with the different formats and structures of data (e.g., text, images, videos), and veracity is concerned with the accuracy and reliability of data, especially in the context of noisy or incomplete datasets.

High velocity is particularly relevant in applications requiring real-time insights, such as stock trading, live sports analytics, and online recommendation engines.
