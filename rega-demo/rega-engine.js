/**
 * ReGA Engine - Pure JavaScript Implementation
 * Graph-based RAG verification with Sinkhorn normalization
 */

import { embeddingEngine } from './embeddings.js';

class ReGAEngine {
    constructor() {
        this.params = {
            sinkhornIterations: 10,
            temperature: 1.0,
            matchSteps: 5,
            threshold: 0.15
        };
        this.metrics = {
            embedTime: 0,
            graphTime: 0,
            sinkhornTime: 0,
            totalTime: 0
        };
    }

    /**
     * Set hyperparameters
     */
    setParams(params) {
        this.params = { ...this.params, ...params };
    }

    /**
     * Sinkhorn-Knopp algorithm for doubly-stochastic normalization
     * @param {number[][]} matrix - Cost/similarity matrix
     * @param {number} iters - Number of iterations
     * @param {number} temperature - Temperature for softmax
     * @returns {number[][]} - Doubly-stochastic matrix
     */
    sinkhorn(matrix, iters = 10, temperature = 1.0) {
        const rows = matrix.length;
        const cols = matrix[0].length;

        // Apply temperature and convert to log domain for stability
        let logAlpha = matrix.map(row =>
            row.map(val => val / (temperature + 1e-10))
        );

        // Subtract max for numerical stability
        const maxVal = Math.max(...logAlpha.flat());
        logAlpha = logAlpha.map(row => row.map(val => val - maxVal));

        // Exponentiate
        let P = logAlpha.map(row =>
            row.map(val => Math.max(Math.exp(val), 1e-10))
        );

        // Sinkhorn iterations
        for (let iter = 0; iter < iters; iter++) {
            // Row normalization
            for (let i = 0; i < rows; i++) {
                const rowSum = P[i].reduce((a, b) => a + b, 0) + 1e-8;
                for (let j = 0; j < cols; j++) {
                    P[i][j] /= rowSum;
                }
            }

            // Column normalization
            for (let j = 0; j < cols; j++) {
                let colSum = 0;
                for (let i = 0; i < rows; i++) {
                    colSum += P[i][j];
                }
                colSum += 1e-8;
                for (let i = 0; i < rows; i++) {
                    P[i][j] /= colSum;
                }
            }
        }

        return P;
    }

    /**
     * Simple GNN-style message passing
     * @param {number[][]} nodeFeatures - Node feature matrix
     * @param {number[][]} adjacency - Adjacency matrix
     * @param {number} steps - Number of message passing steps
     * @returns {number[][]} - Updated node features
     */
    graphEncode(nodeFeatures, adjacency, steps = 2) {
        let H = nodeFeatures.map(row => [...row]);
        const n = H.length;
        const d = H[0].length;

        for (let step = 0; step < steps; step++) {
            const newH = [];

            for (let i = 0; i < n; i++) {
                const newFeatures = new Array(d).fill(0);
                let neighborCount = 0;

                // Aggregate neighbor features
                for (let j = 0; j < n; j++) {
                    if (adjacency[i][j] > 0) {
                        for (let k = 0; k < d; k++) {
                            newFeatures[k] += H[j][k] * adjacency[i][j];
                        }
                        neighborCount += adjacency[i][j];
                    }
                }

                // Combine self and neighbor features (simplified GNN update)
                for (let k = 0; k < d; k++) {
                    if (neighborCount > 0) {
                        // Average of self and mean of neighbors
                        newFeatures[k] = 0.5 * H[i][k] + 0.5 * (newFeatures[k] / neighborCount);
                    } else {
                        newFeatures[k] = H[i][k];
                    }
                }

                // ReLU activation
                newH.push(newFeatures.map(v => Math.max(0, v)));
            }

            H = newH;
        }

        // L2 normalize each node embedding
        return H.map(row => {
            const norm = Math.sqrt(row.reduce((sum, v) => sum + v * v, 0)) + 1e-8;
            return row.map(v => v / norm);
        });
    }

    /**
     * Build adjacency matrix for sequential text
     * Nodes are connected if they are adjacent sentences
     */
    buildAdjacency(numNodes) {
        const adj = [];
        for (let i = 0; i < numNodes; i++) {
            adj.push(new Array(numNodes).fill(0));
            adj[i][i] = 1; // Self-connection
            if (i > 0) adj[i][i - 1] = 1; // Previous sentence
            if (i < numNodes - 1) adj[i][i + 1] = 1; // Next sentence
        }
        return adj;
    }

    /**
     * Compute cosine similarity between two vectors
     */
    cosineSimilarity(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
    }

    /**
     * Compute alignment energy between source and hypothesis graphs
     */
    computeEnergy(sourceEmbeddings, hypEmbeddings, alignmentMatrix) {
        let energy = 0;
        let count = 0;

        for (let i = 0; i < hypEmbeddings.length; i++) {
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                const similarity = this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]);
                // Cost is 1 - similarity, weighted by alignment probability
                energy += alignmentMatrix[i][j] * (1 - similarity);
                count += alignmentMatrix[i][j];
            }
        }

        return count > 0 ? energy / count : 1.0;
    }

    /**
     * Extract ReGA features for classification
     */
    extractFeatures(sourceEmbeddings, hypEmbeddings, alignmentMatrix) {
        const features = [];

        // 1. Mean embedding distance
        const meanSrc = this.meanEmbedding(sourceEmbeddings);
        const meanHyp = this.meanEmbedding(hypEmbeddings);
        features.push(1 - this.cosineSimilarity(meanSrc, meanHyp));

        // 2. Alignment entropy
        let entropy = 0;
        for (const row of alignmentMatrix) {
            for (const p of row) {
                if (p > 1e-8) {
                    entropy -= p * Math.log(p);
                }
            }
        }
        features.push(entropy);

        // 3. Max alignment score (best match)
        const maxAlign = Math.max(...alignmentMatrix.flat());
        features.push(maxAlign);

        // 4. Min row-wise max (worst best-match)
        const rowMaxes = alignmentMatrix.map(row => Math.max(...row));
        features.push(Math.min(...rowMaxes));

        // 5. Diagonal vs off-diagonal ratio (for order preservation)
        const n = Math.min(alignmentMatrix.length, alignmentMatrix[0].length);
        let diagSum = 0, offDiagSum = 0, diagCount = 0, offDiagCount = 0;
        for (let i = 0; i < alignmentMatrix.length; i++) {
            for (let j = 0; j < alignmentMatrix[0].length; j++) {
                if (i < n && j < n && i === j) {
                    diagSum += alignmentMatrix[i][j];
                    diagCount++;
                } else {
                    offDiagSum += alignmentMatrix[i][j];
                    offDiagCount++;
                }
            }
        }
        const diagMean = diagCount > 0 ? diagSum / diagCount : 0;
        const offDiagMean = offDiagCount > 0 ? offDiagSum / offDiagCount : 0;
        features.push(diagMean - offDiagMean);

        return features;
    }

    meanEmbedding(embeddings) {
        const d = embeddings[0].length;
        const mean = new Array(d).fill(0);
        for (const emb of embeddings) {
            for (let i = 0; i < d; i++) {
                mean[i] += emb[i];
            }
        }
        for (let i = 0; i < d; i++) {
            mean[i] /= embeddings.length;
        }
        return mean;
    }

    /**
     * Full verification pipeline
     * @param {string} sourceText - Source/reference text
     * @param {string} hypothesisText - Hypothesis to verify
     * @returns {Object} - Verification results
     */
    async verify(sourceText, hypothesisText) {
        const startTime = performance.now();
        this.metrics = { embedTime: 0, graphTime: 0, sinkhornTime: 0, totalTime: 0 };

        // 1. Split into sentences
        const sourceSentences = embeddingEngine.splitIntoSentences(sourceText);
        const hypSentences = embeddingEngine.splitIntoSentences(hypothesisText);

        // 2. Get embeddings
        const embedStart = performance.now();
        const sourceEmbeddings = await embeddingEngine.embedBatch(sourceSentences);
        const hypEmbeddings = await embeddingEngine.embedBatch(hypSentences);
        this.metrics.embedTime = performance.now() - embedStart;

        // Convert to regular arrays for processing
        const srcEmb = sourceEmbeddings.map(e => Array.from(e));
        const hypEmb = hypEmbeddings.map(e => Array.from(e));

        // 3. Build graphs and encode
        const graphStart = performance.now();
        const srcAdj = this.buildAdjacency(srcEmb.length);
        const hypAdj = this.buildAdjacency(hypEmb.length);

        const srcEncoded = this.graphEncode(srcEmb, srcAdj, this.params.matchSteps);
        const hypEncoded = this.graphEncode(hypEmb, hypAdj, this.params.matchSteps);
        this.metrics.graphTime = performance.now() - graphStart;

        // 4. Compute similarity matrix
        const sinkhornStart = performance.now();
        const similarityMatrix = [];
        for (let i = 0; i < hypEncoded.length; i++) {
            const row = [];
            for (let j = 0; j < srcEncoded.length; j++) {
                row.push(this.cosineSimilarity(hypEncoded[i], srcEncoded[j]));
            }
            similarityMatrix.push(row);
        }

        // 5. Apply Sinkhorn normalization
        const alignmentMatrix = this.sinkhorn(
            similarityMatrix,
            this.params.sinkhornIterations,
            this.params.temperature
        );
        this.metrics.sinkhornTime = performance.now() - sinkhornStart;

        // 6. Compute energy score
        const energy = this.computeEnergy(srcEncoded, hypEncoded, alignmentMatrix);

        // 7. Extract features for detailed analysis
        const features = this.extractFeatures(srcEncoded, hypEncoded, alignmentMatrix);

        // 8. Make verdict
        const isHallucination = energy > this.params.threshold;

        this.metrics.totalTime = performance.now() - startTime;

        return {
            energy,
            isHallucination,
            verdict: isHallucination ? 'FAIL' : 'PASS',
            confidence: isHallucination ? Math.min(energy * 2, 1) : Math.min((1 - energy) * 2, 1),
            metrics: { ...this.metrics },
            details: {
                sourceSentences,
                hypSentences,
                sourceEmbeddings: srcEncoded,
                hypEmbeddings: hypEncoded,
                alignmentMatrix,
                features
            }
        };
    }
}

// Export singleton instance
export const regaEngine = new ReGAEngine();
export default regaEngine;
