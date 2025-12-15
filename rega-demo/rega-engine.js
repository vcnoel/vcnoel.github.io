/**
 * ReGA Engine - Proper Implementation per Paper
 * 
 * Based on: "ReGA: Zero-Cost Graph Alignment for Structural Hallucination Detection"
 * 
 * Key principles from paper:
 * - Energy is CONTINUOUS: 0.00 = faithful, higher = hallucination
 * - Structural energy: E = ||A_S - P^T A_H P||_F^2
 * - Sinkhorn normalization for soft permutation
 */

import { embeddingEngine } from './embeddings.js';

class ReGAEngine {
    constructor() {
        this.params = {
            sinkhornIterations: 10,
            temperature: 1.0,
            matchSteps: 5,
            threshold: 0.15,  // From paper: energy > 0.15 = hallucination
        };

        this.metrics = {
            embedTime: 0,
            featureTime: 0,
            deepRegaTime: 0,
            totalTime: 0,
            stage: 'none'
        };
    }

    setParams(params) {
        this.params = { ...this.params, ...params };
    }

    // =========================================================================
    // SINKHORN NORMALIZATION (from paper Section 3.2)
    // =========================================================================

    /**
     * Sinkhorn-Knopp algorithm for doubly-stochastic normalization
     * Produces soft permutation matrix P
     */
    sinkhorn(simMatrix, iters = 10, temperature = 1.0) {
        const m = simMatrix.length;
        const n = simMatrix[0].length;

        // Initialize with scaled similarities
        let P = simMatrix.map(row =>
            row.map(val => Math.exp(val / temperature))
        );

        // Alternate row and column normalization
        for (let iter = 0; iter < iters; iter++) {
            // Row normalization
            for (let i = 0; i < m; i++) {
                const rowSum = P[i].reduce((a, b) => a + b, 0) + 1e-10;
                P[i] = P[i].map(v => v / rowSum);
            }
            // Column normalization
            for (let j = 0; j < n; j++) {
                let colSum = 1e-10;
                for (let i = 0; i < m; i++) colSum += P[i][j];
                for (let i = 0; i < m; i++) P[i][j] /= colSum;
            }
        }

        return P;
    }

    // =========================================================================
    // STRUCTURAL ENERGY (from paper Equation in Section 3.2)
    // E = ||A_S - P^T A_H P||_F^2
    // =========================================================================

    /**
     * Build adjacency matrix for sequential text (sentences as nodes)
     * Adjacent sentences are connected
     */
    buildAdjacency(numNodes) {
        const adj = [];
        for (let i = 0; i < numNodes; i++) {
            adj.push(new Array(numNodes).fill(0));
            adj[i][i] = 1;  // Self-loop
            if (i > 0) adj[i][i - 1] = 1;  // Previous
            if (i < numNodes - 1) adj[i][i + 1] = 1;  // Next
        }
        return adj;
    }

    /**
     * Compute structural energy: how well hypothesis structure aligns with source
     * Lower energy = better alignment = more faithful
     */
    computeStructuralEnergy(srcAdj, hypAdj, P) {
        const n = srcAdj.length;
        const m = hypAdj.length;

        // Compute P^T A_H P (transformed hypothesis adjacency)
        // Then measure Frobenius distance to A_S

        let energy = 0;
        let count = 0;

        // For each pair of source nodes, compare their edge to aligned hypothesis edges
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                // Compute (P^T A_H P)[i][j]
                let alignedVal = 0;
                for (let hi = 0; hi < m; hi++) {
                    for (let hj = 0; hj < m; hj++) {
                        alignedVal += P[hi][i] * hypAdj[hi][hj] * P[hj][j];
                    }
                }

                const diff = srcAdj[i][j] - alignedVal;
                energy += diff * diff;
                count++;
            }
        }

        // Normalize by number of comparisons
        return count > 0 ? Math.sqrt(energy / count) : 0;
    }

    /**
     * Compute alignment energy: weighted cost under soft permutation
     */
    computeAlignmentEnergy(simMatrix, P) {
        let energy = 0;
        const m = P.length;
        const n = P[0].length;

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                // Cost = 1 - similarity, weighted by permutation
                const cost = 1 - simMatrix[i][j];
                energy += P[i][j] * cost;
            }
        }

        return energy;
    }

    // =========================================================================
    // FEATURE EXTRACTION (for explanations, not primary energy)
    // =========================================================================

    extractNumbers(text) {
        const numbers = [];
        const regex = /\b\d+(?:\.\d+)?\b/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            numbers.push(parseFloat(match[0]));
        }
        return numbers;
    }

    /**
     * Detect factual issues for explanations
     * Returns explanations array, not used for energy calculation
     */
    detectFactualIssues(sourceText, hypText) {
        const explanations = [];
        const srcLower = sourceText.toLowerCase();
        const hypLower = hypText.toLowerCase();

        // Number differences
        const srcNums = this.extractNumbers(sourceText);
        const hypNums = this.extractNumbers(hypText);

        for (const hypNum of hypNums) {
            const hasMatch = srcNums.some(srcNum =>
                Math.abs(hypNum - srcNum) / Math.max(srcNum, 1) < 0.05
            );
            if (!hasMatch && srcNums.length > 0) {
                const closest = srcNums.reduce((a, b) =>
                    Math.abs(b - hypNum) < Math.abs(a - hypNum) ? b : a
                );
                explanations.push({
                    type: 'number',
                    icon: '🔢',
                    text: `Number changed: "${hypNum}" (expected ~${closest})`
                });
            }
        }

        // Antonym detection
        const antonymPairs = [
            ['tallest', 'smallest'], ['largest', 'smallest'], ['highest', 'lowest'],
            ['best', 'worst'], ['first', 'last'], ['increase', 'decrease'],
            ['always', 'never'], ['all', 'none'], ['true', 'false']
        ];

        for (const [word1, word2] of antonymPairs) {
            if (srcLower.includes(word1) && hypLower.includes(word2)) {
                explanations.push({
                    type: 'antonym',
                    icon: '⚠️',
                    text: `Contradiction: "${word1}" → "${word2}"`
                });
            } else if (srcLower.includes(word2) && hypLower.includes(word1)) {
                explanations.push({
                    type: 'antonym',
                    icon: '⚠️',
                    text: `Contradiction: "${word2}" → "${word1}"`
                });
            }
        }

        return explanations;
    }

    // =========================================================================
    // COSINE SIMILARITY
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

    // =========================================================================
    // MAIN VERIFICATION (Following paper methodology)
    // =========================================================================

    async verify(sourceText, hypothesisText) {
        const startTime = performance.now();
        this.metrics = {
            embedTime: 0,
            featureTime: 0,
            deepRegaTime: 0,
            totalTime: 0,
            stage: 'none'
        };

        // Split text into sentences
        const sourceSentences = embeddingEngine.splitIntoSentences(sourceText);
        const hypSentences = embeddingEngine.splitIntoSentences(hypothesisText);

        // Get embeddings
        const embedStart = performance.now();
        const sourceEmbeddings = await embeddingEngine.embedBatch(sourceSentences);
        const hypEmbeddings = await embeddingEngine.embedBatch(hypSentences);
        this.metrics.embedTime = performance.now() - embedStart;

        // Feature extraction for explanations
        const featureStart = performance.now();
        const explanations = this.detectFactualIssues(sourceText, hypothesisText);
        this.metrics.featureTime = performance.now() - featureStart;

        // Deep ReGA: Compute similarity matrix
        const deepStart = performance.now();

        const simMatrix = [];
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const row = [];
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                row.push(this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            simMatrix.push(row);
        }

        // Sinkhorn normalization for soft permutation
        const P = this.sinkhorn(
            simMatrix,
            this.params.sinkhornIterations,
            this.params.temperature
        );

        // Build adjacency matrices
        const srcAdj = this.buildAdjacency(sourceEmbeddings.length);
        const hypAdj = this.buildAdjacency(hypEmbeddings.length);

        // Compute energies (following paper)
        const structuralEnergy = this.computeStructuralEnergy(srcAdj, hypAdj, P);
        const alignmentEnergy = this.computeAlignmentEnergy(simMatrix, P);

        // Combined energy (paper uses structural primarily)
        // Weight: 60% structural, 40% alignment
        let finalEnergy = 0.6 * structuralEnergy + 0.4 * alignmentEnergy;

        // Add small penalty for detected factual issues (explanatory, not dominant)
        const factualPenalty = Math.min(explanations.length * 0.05, 0.2);
        finalEnergy += factualPenalty;

        // Clamp to reasonable range [0, 1] for display
        finalEnergy = Math.max(0, Math.min(1, finalEnergy));

        this.metrics.deepRegaTime = performance.now() - deepStart;

        // Decision based on threshold
        const isHallucination = finalEnergy > this.params.threshold;
        const confidence = isHallucination
            ? Math.min((finalEnergy - this.params.threshold) / 0.3 + 0.5, 1.0)
            : Math.min((this.params.threshold - finalEnergy) / this.params.threshold + 0.5, 1.0);

        // Add explanations for low similarity alignments
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const maxSim = Math.max(...simMatrix[i]);
            if (maxSim < 0.6) {
                explanations.push({
                    type: 'alignment',
                    icon: '🔗',
                    text: `Weak match for sentence ${i + 1} (${(maxSim * 100).toFixed(0)}% sim)`
                });
            }
        }

        let stage = 'Deep ReGA';
        let stageReason = 'Sinkhorn alignment + structural energy';

        if (explanations.some(e => e.type === 'antonym')) {
            stage = 'Factual Analysis';
            stageReason = 'Semantic contradiction detected';
        } else if (explanations.some(e => e.type === 'number')) {
            stage = 'Factual Analysis';
            stageReason = 'Numerical mismatch detected';
        }

        this.metrics.stage = stage.toLowerCase().replace(/\s+/g, '-');
        this.metrics.totalTime = performance.now() - startTime;

        return {
            energy: finalEnergy,
            isHallucination,
            verdict: isHallucination ? 'FAIL' : 'PASS',
            confidence,
            stage,
            stageReason,
            explanations,
            metrics: { ...this.metrics },
            details: {
                sourceSentences,
                hypSentences,
                sourceEmbeddings: sourceEmbeddings.map(e => Array.from(e)),
                hypEmbeddings: hypEmbeddings.map(e => Array.from(e)),
                alignmentMatrix: P,
                structuralEnergy,
                alignmentEnergy,
                simMatrix
            }
        };
    }
}

export const regaEngine = new ReGAEngine();
export default regaEngine;
