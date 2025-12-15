/**
 * ReGA Engine - Cascading Architecture
 * 
 * Stage 1: Feature-based ReGA (interpretable features + logistic regression)
 * Stage 2: Deep ReGA (GNN + Sinkhorn normalization for directional hallucinations)
 * 
 * Based on: "ReGA: Zero-Cost Graph Alignment for Structural Hallucination Detection"
 */

import { embeddingEngine } from './embeddings.js';

class ReGAEngine {
    constructor() {
        this.params = {
            // Stage 1: Feature ReGA
            featureThreshold: 0.5,
            ambiguityMargin: 0.15,  // If score is within this margin of threshold, escalate to Deep ReGA

            // Stage 2: Deep ReGA
            sinkhornIterations: 10,
            temperature: 1.0,
            matchSteps: 5,
            energyThreshold: 0.15,

            // Cascade control
            useCascade: true,  // If false, use Deep ReGA only
        };

        this.metrics = {
            embedTime: 0,
            featureTime: 0,
            deepRegaTime: 0,
            totalTime: 0,
            stage: 'none'  // 'feature', 'deep', or 'cascade'
        };

        // Logistic regression weights (trained on synthetic data)
        // Features: [meanDist, swapIndicator, maxCost, minCost, stdCost, diagVsOffDiag, entropyProxy]
        this.featureWeights = [2.1, 1.8, 1.5, -0.8, 0.9, 1.2, 0.6];
        this.featureBias = -1.5;
    }

    /**
     * Set hyperparameters
     */
    setParams(params) {
        this.params = { ...this.params, ...params };
    }

    // =========================================================================
    // STAGE 1: Feature-based ReGA
    // =========================================================================

    /**
     * Extract hand-crafted features for Feature ReGA
     * Based on Section 3.1 of the paper
     */
    extractFeatures(sourceEmbeddings, hypEmbeddings) {
        const features = [];

        // 1. Mean embedding distance (graph-level semantic distance)
        const meanSrc = this.meanEmbedding(sourceEmbeddings);
        const meanHyp = this.meanEmbedding(hypEmbeddings);
        const meanDist = 1 - this.cosineSimilarity(meanSrc, meanHyp);
        features.push(meanDist);

        // 2. Build cost matrix (1 - similarity)
        const costMatrix = [];
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const row = [];
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                row.push(1 - this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            costMatrix.push(row);
        }

        // 3. Swap Indicator (Equation 1 in paper)
        // Measures if distinct roles are confused
        let swapSum = 0;
        for (let j = 0; j < hypEmbeddings.length; j++) {
            const minCost = Math.min(...costMatrix[j]);
            const otherCosts = costMatrix[j].filter((_, idx) => costMatrix[j][idx] !== minCost);
            const meanOtherCost = otherCosts.length > 0
                ? otherCosts.reduce((a, b) => a + b, 0) / otherCosts.length
                : minCost;
            const gap = meanOtherCost - minCost;
            swapSum += gap > 0 ? 1 / (gap + 0.01) : 100;  // Inverse gap
        }
        features.push(swapSum / hypEmbeddings.length);

        // 4. Cost matrix statistics
        const flatCosts = costMatrix.flat();
        features.push(Math.max(...flatCosts));  // maxCost
        features.push(Math.min(...flatCosts));  // minCost
        features.push(this.std(flatCosts));     // stdCost

        // 5. Diagonal vs off-diagonal (order preservation indicator)
        const n = Math.min(sourceEmbeddings.length, hypEmbeddings.length);
        let diagSum = 0, offDiagSum = 0, diagCount = 0, offDiagCount = 0;
        for (let i = 0; i < costMatrix.length; i++) {
            for (let j = 0; j < costMatrix[0].length; j++) {
                if (i < n && j < n && i === j) {
                    diagSum += costMatrix[i][j];
                    diagCount++;
                } else {
                    offDiagSum += costMatrix[i][j];
                    offDiagCount++;
                }
            }
        }
        const diagMean = diagCount > 0 ? diagSum / diagCount : 0;
        const offDiagMean = offDiagCount > 0 ? offDiagSum / offDiagCount : 0;
        features.push(diagMean - offDiagMean);

        // 6. Entropy proxy (alignment ambiguity)
        let entropyProxy = 0;
        for (let j = 0; j < hypEmbeddings.length; j++) {
            const similarities = costMatrix[j].map(c => Math.exp(-c));
            const sum = similarities.reduce((a, b) => a + b, 0);
            const probs = similarities.map(s => s / sum);
            entropyProxy += probs.reduce((e, p) => e - (p > 0 ? p * Math.log(p) : 0), 0);
        }
        features.push(entropyProxy / hypEmbeddings.length);

        return features;
    }

    /**
     * Feature-based ReGA classification
     * Returns: { score, prediction, confidence, isAmbiguous }
     */
    featureReGA(sourceEmbeddings, hypEmbeddings) {
        const features = this.extractFeatures(sourceEmbeddings, hypEmbeddings);

        // Logistic regression: sigmoid(w·x + b)
        let logit = this.featureBias;
        for (let i = 0; i < features.length; i++) {
            logit += this.featureWeights[i] * features[i];
        }
        const score = 1 / (1 + Math.exp(-logit));

        const prediction = score > this.params.featureThreshold;
        const distFromThreshold = Math.abs(score - this.params.featureThreshold);
        const isAmbiguous = distFromThreshold < this.params.ambiguityMargin;
        const confidence = isAmbiguous ? 0.5 : Math.min(distFromThreshold * 2 + 0.5, 1.0);

        return {
            score,
            prediction,
            confidence,
            isAmbiguous,
            features
        };
    }

    // =========================================================================
    // STAGE 2: Deep ReGA (Neural Graph Matching)
    // =========================================================================

    /**
     * Sinkhorn-Knopp algorithm for doubly-stochastic normalization
     */
    sinkhorn(matrix, iters = 10, temperature = 1.0) {
        const rows = matrix.length;
        const cols = matrix[0].length;

        let logAlpha = matrix.map(row =>
            row.map(val => val / (temperature + 1e-10))
        );

        const maxVal = Math.max(...logAlpha.flat());
        logAlpha = logAlpha.map(row => row.map(val => val - maxVal));

        let P = logAlpha.map(row =>
            row.map(val => Math.max(Math.exp(val), 1e-10))
        );

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
     * GNN-style message passing (simplified)
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

                for (let j = 0; j < n; j++) {
                    if (adjacency[i][j] > 0) {
                        for (let k = 0; k < d; k++) {
                            newFeatures[k] += H[j][k] * adjacency[i][j];
                        }
                        neighborCount += adjacency[i][j];
                    }
                }

                for (let k = 0; k < d; k++) {
                    if (neighborCount > 0) {
                        newFeatures[k] = 0.5 * H[i][k] + 0.5 * (newFeatures[k] / neighborCount);
                    } else {
                        newFeatures[k] = H[i][k];
                    }
                }

                newH.push(newFeatures.map(v => Math.max(0, v)));
            }

            H = newH;
        }

        return H.map(row => {
            const norm = Math.sqrt(row.reduce((sum, v) => sum + v * v, 0)) + 1e-8;
            return row.map(v => v / norm);
        });
    }

    /**
     * Build adjacency matrix for sequential text
     */
    buildAdjacency(numNodes) {
        const adj = [];
        for (let i = 0; i < numNodes; i++) {
            adj.push(new Array(numNodes).fill(0));
            adj[i][i] = 1;
            if (i > 0) adj[i][i - 1] = 1;
            if (i < numNodes - 1) adj[i][i + 1] = 1;
        }
        return adj;
    }

    /**
     * Compute structural energy (Equation from paper)
     * E = ||A_S - P^T A_H P||_F^2
     */
    computeStructuralEnergy(srcAdj, hypAdj, permutation) {
        const n = srcAdj.length;
        const m = hypAdj.length;

        // Compute P^T A_H P (transformed hypothesis adjacency)
        // This checks if edge structure is preserved under alignment
        let energy = 0;
        let count = 0;

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                let alignedValue = 0;
                for (let hi = 0; hi < m; hi++) {
                    for (let hj = 0; hj < m; hj++) {
                        alignedValue += permutation[hi][i] * hypAdj[hi][hj] * permutation[hj][j];
                    }
                }
                const diff = srcAdj[i][j] - alignedValue;
                energy += diff * diff;
                count++;
            }
        }

        return count > 0 ? Math.sqrt(energy / count) : 0;
    }

    /**
     * Deep ReGA verification
     */
    deepReGA(sourceEmbeddings, hypEmbeddings) {
        const srcEmb = sourceEmbeddings.map(e => Array.from(e));
        const hypEmb = hypEmbeddings.map(e => Array.from(e));

        // Build graphs
        const srcAdj = this.buildAdjacency(srcEmb.length);
        const hypAdj = this.buildAdjacency(hypEmb.length);

        // GNN encoding
        const srcEncoded = this.graphEncode(srcEmb, srcAdj, this.params.matchSteps);
        const hypEncoded = this.graphEncode(hypEmb, hypAdj, this.params.matchSteps);

        // Compute similarity matrix
        const similarityMatrix = [];
        for (let i = 0; i < hypEncoded.length; i++) {
            const row = [];
            for (let j = 0; j < srcEncoded.length; j++) {
                row.push(this.cosineSimilarity(hypEncoded[i], srcEncoded[j]));
            }
            similarityMatrix.push(row);
        }

        // Sinkhorn normalization for soft permutation
        const permutation = this.sinkhorn(
            similarityMatrix,
            this.params.sinkhornIterations,
            this.params.temperature
        );

        // Compute structural energy
        const energy = this.computeStructuralEnergy(srcAdj, hypAdj, permutation);

        // Also compute alignment-based energy
        let alignmentEnergy = 0;
        for (let i = 0; i < hypEncoded.length; i++) {
            for (let j = 0; j < srcEncoded.length; j++) {
                const similarity = this.cosineSimilarity(hypEncoded[i], srcEncoded[j]);
                alignmentEnergy += permutation[i][j] * (1 - similarity);
            }
        }

        // Combined energy (structural + alignment)
        const combinedEnergy = 0.6 * energy + 0.4 * alignmentEnergy;

        const prediction = combinedEnergy > this.params.energyThreshold;
        const confidence = prediction
            ? Math.min(combinedEnergy / 0.3, 1.0)
            : Math.min((this.params.energyThreshold - combinedEnergy) / this.params.energyThreshold + 0.5, 1.0);

        return {
            energy: combinedEnergy,
            structuralEnergy: energy,
            alignmentEnergy,
            prediction,
            confidence,
            permutation,
            srcEncoded,
            hypEncoded
        };
    }

    // =========================================================================
    // CASCADE PIPELINE
    // =========================================================================

    /**
     * Full cascading verification pipeline
     */
    async verify(sourceText, hypothesisText) {
        const startTime = performance.now();
        this.metrics = {
            embedTime: 0,
            featureTime: 0,
            deepRegaTime: 0,
            totalTime: 0,
            stage: 'none'
        };

        // Split into sentences
        const sourceSentences = embeddingEngine.splitIntoSentences(sourceText);
        const hypSentences = embeddingEngine.splitIntoSentences(hypothesisText);

        // Get embeddings
        const embedStart = performance.now();
        const sourceEmbeddings = await embeddingEngine.embedBatch(sourceSentences);
        const hypEmbeddings = await embeddingEngine.embedBatch(hypSentences);
        this.metrics.embedTime = performance.now() - embedStart;

        let result;

        if (this.params.useCascade) {
            // STAGE 1: Feature-based ReGA
            const featureStart = performance.now();
            const featureResult = this.featureReGA(sourceEmbeddings, hypEmbeddings);
            this.metrics.featureTime = performance.now() - featureStart;

            if (!featureResult.isAmbiguous) {
                // Feature ReGA is confident - return immediately
                this.metrics.stage = 'feature';
                result = {
                    energy: featureResult.score,
                    isHallucination: featureResult.prediction,
                    verdict: featureResult.prediction ? 'FAIL' : 'PASS',
                    confidence: featureResult.confidence,
                    stage: 'Feature ReGA',
                    stageReason: 'High confidence from interpretable features',
                    featureResult
                };
            } else {
                // Ambiguous - escalate to Deep ReGA
                const deepStart = performance.now();
                const deepResult = this.deepReGA(sourceEmbeddings, hypEmbeddings);
                this.metrics.deepRegaTime = performance.now() - deepStart;
                this.metrics.stage = 'cascade';

                result = {
                    energy: deepResult.energy,
                    isHallucination: deepResult.prediction,
                    verdict: deepResult.prediction ? 'FAIL' : 'PASS',
                    confidence: deepResult.confidence,
                    stage: 'Deep ReGA (cascaded)',
                    stageReason: 'Feature ReGA ambiguous, used neural graph matching',
                    featureResult,
                    deepResult
                };
            }
        } else {
            // Deep ReGA only mode
            const deepStart = performance.now();
            const deepResult = this.deepReGA(sourceEmbeddings, hypEmbeddings);
            this.metrics.deepRegaTime = performance.now() - deepStart;
            this.metrics.stage = 'deep';

            result = {
                energy: deepResult.energy,
                isHallucination: deepResult.prediction,
                verdict: deepResult.prediction ? 'FAIL' : 'PASS',
                confidence: deepResult.confidence,
                stage: 'Deep ReGA',
                stageReason: 'Direct neural graph matching',
                deepResult
            };
        }

        this.metrics.totalTime = performance.now() - startTime;

        return {
            ...result,
            metrics: { ...this.metrics },
            details: {
                sourceSentences,
                hypSentences,
                sourceEmbeddings: sourceEmbeddings.map(e => Array.from(e)),
                hypEmbeddings: hypEmbeddings.map(e => Array.from(e)),
                alignmentMatrix: result.deepResult?.permutation || this.computeSimpleAlignment(sourceEmbeddings, hypEmbeddings)
            }
        };
    }

    /**
     * Simple alignment for visualization when Feature ReGA is used
     */
    computeSimpleAlignment(sourceEmbeddings, hypEmbeddings) {
        const matrix = [];
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const row = [];
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                row.push(this.cosineSimilarity(
                    Array.from(hypEmbeddings[i]),
                    Array.from(sourceEmbeddings[j])
                ));
            }
            // Softmax normalization
            const maxVal = Math.max(...row);
            const expRow = row.map(v => Math.exp((v - maxVal) / 0.5));
            const sum = expRow.reduce((a, b) => a + b, 0);
            matrix.push(expRow.map(v => v / sum));
        }
        return matrix;
    }

    // =========================================================================
    // UTILITY FUNCTIONS
    // =========================================================================

    cosineSimilarity(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
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

    std(arr) {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length;
        return Math.sqrt(variance);
    }
}

export const regaEngine = new ReGAEngine();
export default regaEngine;
