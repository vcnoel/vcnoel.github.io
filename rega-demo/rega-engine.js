/**
 * ReGA Engine - Cascading Architecture
 * 
 * Stage 1: Feature-based ReGA (interpretable features + logistic regression)
 * Stage 2: Deep ReGA (GNN + Sinkhorn normalization for directional hallucinations)
 * 
 * Enhanced with text-level factual analysis for catching numerical and semantic contradictions
 * 
 * Based on: "ReGA: Zero-Cost Graph Alignment for Structural Hallucination Detection"
 */

import { embeddingEngine } from './embeddings.js';

class ReGAEngine {
    constructor() {
        this.params = {
            // Stage 1: Feature ReGA
            featureThreshold: 0.5,
            ambiguityMargin: 0.15,

            // Stage 2: Deep ReGA
            sinkhornIterations: 10,
            temperature: 1.0,
            matchSteps: 5,
            energyThreshold: 0.15,

            // Cascade control
            useCascade: true,
        };

        this.metrics = {
            embedTime: 0,
            featureTime: 0,
            deepRegaTime: 0,
            totalTime: 0,
            stage: 'none'
        };

        // Logistic regression weights for combined features
        // [meanDist, swapIndicator, maxCost, minCost, stdCost, diagVsOffDiag, entropyProxy, 
        //  numberMismatch, antonymScore, entityMismatch, lengthRatio]
        this.featureWeights = [2.0, 1.5, 1.2, -0.6, 0.8, 1.0, 0.5, 3.5, 4.0, 2.5, 0.3];
        this.featureBias = -2.0;

        // Antonym pairs for detecting semantic contradictions
        this.antonyms = new Map([
            ['tallest', 'smallest'], ['smallest', 'tallest'],
            ['largest', 'smallest'], ['biggest', 'smallest'],
            ['highest', 'lowest'], ['lowest', 'highest'],
            ['first', 'last'], ['last', 'first'],
            ['best', 'worst'], ['worst', 'best'],
            ['increase', 'decrease'], ['decrease', 'increase'],
            ['rise', 'fall'], ['fall', 'rise'],
            ['gain', 'loss'], ['loss', 'gain'],
            ['success', 'failure'], ['failure', 'success'],
            ['true', 'false'], ['false', 'true'],
            ['positive', 'negative'], ['negative', 'positive'],
            ['before', 'after'], ['after', 'before'],
            ['above', 'below'], ['below', 'above'],
            ['more', 'less'], ['less', 'more'],
            ['always', 'never'], ['never', 'always'],
            ['all', 'none'], ['none', 'all'],
            ['acquired', 'sold'], ['bought', 'sold'],
            ['created', 'destroyed'], ['built', 'demolished'],
        ]);
    }

    setParams(params) {
        this.params = { ...this.params, ...params };
    }

    // =========================================================================
    // TEXT-LEVEL ANALYSIS (for catching factual hallucinations)
    // =========================================================================

    /**
     * Extract numbers from text
     */
    extractNumbers(text) {
        const numbers = [];
        // Match various number formats: integers, decimals, with commas
        const regex = /\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\b/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            const numStr = match[0].replace(/,/g, '');
            numbers.push(parseFloat(numStr));
        }
        return numbers;
    }

    /**
     * Calculate number mismatch score
     * Returns a score indicating how much numbers differ
     */
    calculateNumberMismatch(sourceText, hypText) {
        const sourceNums = this.extractNumbers(sourceText);
        const hypNums = this.extractNumbers(hypText);

        if (sourceNums.length === 0 && hypNums.length === 0) {
            return 0;  // No numbers to compare
        }

        if (sourceNums.length === 0 || hypNums.length === 0) {
            return 0.3;  // One has numbers, other doesn't - mild mismatch
        }

        // For each hypothesis number, find if there's a matching source number
        let mismatchScore = 0;
        let comparisons = 0;

        for (const hypNum of hypNums) {
            let bestMatch = Infinity;
            for (const srcNum of sourceNums) {
                // Calculate relative difference
                const diff = Math.abs(hypNum - srcNum) / (Math.max(Math.abs(srcNum), 1));
                bestMatch = Math.min(bestMatch, diff);
            }

            // If best match is more than 5% different, it's a mismatch
            if (bestMatch > 0.05) {
                mismatchScore += Math.min(bestMatch, 1.0);
            }
            comparisons++;
        }

        return comparisons > 0 ? mismatchScore / comparisons : 0;
    }

    /**
     * Detect antonym usage (semantic contradictions)
     */
    detectAntonyms(sourceText, hypText) {
        const sourceWords = sourceText.toLowerCase().split(/\W+/).filter(w => w.length > 2);
        const hypWords = hypText.toLowerCase().split(/\W+/).filter(w => w.length > 2);

        let antonymCount = 0;

        for (const srcWord of sourceWords) {
            const antonym = this.antonyms.get(srcWord);
            if (antonym && hypWords.includes(antonym)) {
                antonymCount++;
            }
        }

        return Math.min(antonymCount * 0.5, 1.0);  // Cap at 1.0
    }

    /**
     * Extract key entities (proper nouns, capitalized words)
     */
    extractEntities(text) {
        const entities = new Set();
        // Match capitalized words that aren't at sentence start
        const words = text.split(/\s+/);
        for (let i = 0; i < words.length; i++) {
            const word = words[i].replace(/[^a-zA-Z]/g, '');
            // Consider it an entity if capitalized and not at sentence start
            if (word.length > 2 && /^[A-Z]/.test(word)) {
                entities.add(word.toLowerCase());
            }
        }
        return entities;
    }

    /**
     * Calculate entity mismatch
     */
    calculateEntityMismatch(sourceText, hypText) {
        const sourceEntities = this.extractEntities(sourceText);
        const hypEntities = this.extractEntities(hypText);

        if (sourceEntities.size === 0 && hypEntities.size === 0) {
            return 0;
        }

        // Find entities in hypothesis not in source
        let newEntities = 0;
        for (const entity of hypEntities) {
            if (!sourceEntities.has(entity)) {
                newEntities++;
            }
        }

        // Find entities in source missing from hypothesis  
        let missingEntities = 0;
        for (const entity of sourceEntities) {
            if (!hypEntities.has(entity)) {
                missingEntities++;
            }
        }

        const totalEntities = Math.max(sourceEntities.size, hypEntities.size);
        return totalEntities > 0 ? (newEntities + missingEntities * 0.5) / (totalEntities + 1) : 0;
    }

    /**
     * Compute text-level features
     */
    computeTextFeatures(sourceText, hypText) {
        return {
            numberMismatch: this.calculateNumberMismatch(sourceText, hypText),
            antonymScore: this.detectAntonyms(sourceText, hypText),
            entityMismatch: this.calculateEntityMismatch(sourceText, hypText),
            lengthRatio: Math.abs(1 - hypText.length / Math.max(sourceText.length, 1))
        };
    }

    // =========================================================================
    // STAGE 1: Feature-based ReGA
    // =========================================================================

    extractFeatures(sourceEmbeddings, hypEmbeddings, textFeatures = null) {
        const features = [];

        // 1. Mean embedding distance
        const meanSrc = this.meanEmbedding(sourceEmbeddings);
        const meanHyp = this.meanEmbedding(hypEmbeddings);
        const meanDist = 1 - this.cosineSimilarity(meanSrc, meanHyp);
        features.push(meanDist);

        // 2. Build cost matrix
        const costMatrix = [];
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const row = [];
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                row.push(1 - this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            costMatrix.push(row);
        }

        // 3. Swap Indicator
        let swapSum = 0;
        for (let j = 0; j < hypEmbeddings.length; j++) {
            const minCost = Math.min(...costMatrix[j]);
            const otherCosts = costMatrix[j].filter((_, idx) => costMatrix[j][idx] !== minCost);
            const meanOtherCost = otherCosts.length > 0
                ? otherCosts.reduce((a, b) => a + b, 0) / otherCosts.length
                : minCost;
            const gap = meanOtherCost - minCost;
            swapSum += gap > 0 ? 1 / (gap + 0.01) : 100;
        }
        features.push(swapSum / hypEmbeddings.length);

        // 4. Cost matrix statistics
        const flatCosts = costMatrix.flat();
        features.push(Math.max(...flatCosts));
        features.push(Math.min(...flatCosts));
        features.push(this.std(flatCosts));

        // 5. Diagonal vs off-diagonal
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
        features.push((diagCount > 0 ? diagSum / diagCount : 0) - (offDiagCount > 0 ? offDiagSum / offDiagCount : 0));

        // 6. Entropy proxy
        let entropyProxy = 0;
        for (let j = 0; j < hypEmbeddings.length; j++) {
            const similarities = costMatrix[j].map(c => Math.exp(-c));
            const sum = similarities.reduce((a, b) => a + b, 0);
            const probs = similarities.map(s => s / sum);
            entropyProxy += probs.reduce((e, p) => e - (p > 0 ? p * Math.log(p) : 0), 0);
        }
        features.push(entropyProxy / hypEmbeddings.length);

        // 7-10. Text-level features (crucial for factual hallucinations!)
        if (textFeatures) {
            features.push(textFeatures.numberMismatch);
            features.push(textFeatures.antonymScore);
            features.push(textFeatures.entityMismatch);
            features.push(textFeatures.lengthRatio);
        } else {
            features.push(0, 0, 0, 0);
        }

        return features;
    }

    featureReGA(sourceEmbeddings, hypEmbeddings, textFeatures = null) {
        const features = this.extractFeatures(sourceEmbeddings, hypEmbeddings, textFeatures);

        // Logistic regression
        let logit = this.featureBias;
        for (let i = 0; i < Math.min(features.length, this.featureWeights.length); i++) {
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
            features,
            textFeatures
        };
    }

    // =========================================================================
    // STAGE 2: Deep ReGA (Neural Graph Matching)
    // =========================================================================

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
            for (let i = 0; i < rows; i++) {
                const rowSum = P[i].reduce((a, b) => a + b, 0) + 1e-8;
                for (let j = 0; j < cols; j++) {
                    P[i][j] /= rowSum;
                }
            }
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

    computeStructuralEnergy(srcAdj, hypAdj, permutation) {
        const n = srcAdj.length;
        const m = hypAdj.length;

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

    deepReGA(sourceEmbeddings, hypEmbeddings, textFeatures = null) {
        const srcEmb = sourceEmbeddings.map(e => Array.from(e));
        const hypEmb = hypEmbeddings.map(e => Array.from(e));

        const srcAdj = this.buildAdjacency(srcEmb.length);
        const hypAdj = this.buildAdjacency(hypEmb.length);

        const srcEncoded = this.graphEncode(srcEmb, srcAdj, this.params.matchSteps);
        const hypEncoded = this.graphEncode(hypEmb, hypAdj, this.params.matchSteps);

        const similarityMatrix = [];
        for (let i = 0; i < hypEncoded.length; i++) {
            const row = [];
            for (let j = 0; j < srcEncoded.length; j++) {
                row.push(this.cosineSimilarity(hypEncoded[i], srcEncoded[j]));
            }
            similarityMatrix.push(row);
        }

        const permutation = this.sinkhorn(
            similarityMatrix,
            this.params.sinkhornIterations,
            this.params.temperature
        );

        const structuralEnergy = this.computeStructuralEnergy(srcAdj, hypAdj, permutation);

        let alignmentEnergy = 0;
        for (let i = 0; i < hypEncoded.length; i++) {
            for (let j = 0; j < srcEncoded.length; j++) {
                const similarity = this.cosineSimilarity(hypEncoded[i], srcEncoded[j]);
                alignmentEnergy += permutation[i][j] * (1 - similarity);
            }
        }

        // Include text features in the energy calculation
        let textPenalty = 0;
        if (textFeatures) {
            textPenalty = (
                textFeatures.numberMismatch * 0.5 +
                textFeatures.antonymScore * 0.6 +
                textFeatures.entityMismatch * 0.3
            );
        }

        // Combined energy: structural + alignment + text-level
        const combinedEnergy = 0.3 * structuralEnergy + 0.3 * alignmentEnergy + 0.4 * textPenalty;

        const prediction = combinedEnergy > this.params.energyThreshold;
        const confidence = prediction
            ? Math.min(combinedEnergy / 0.3, 1.0)
            : Math.min((this.params.energyThreshold - combinedEnergy) / this.params.energyThreshold + 0.5, 1.0);

        return {
            energy: combinedEnergy,
            structuralEnergy,
            alignmentEnergy,
            textPenalty,
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

    async verify(sourceText, hypothesisText) {
        const startTime = performance.now();
        this.metrics = {
            embedTime: 0,
            featureTime: 0,
            deepRegaTime: 0,
            totalTime: 0,
            stage: 'none'
        };

        // Compute text-level features FIRST (before embeddings)
        const textFeatures = this.computeTextFeatures(sourceText, hypothesisText);

        // Quick rejection for obvious factual mismatches
        const obviousHallucination =
            textFeatures.numberMismatch > 0.3 ||
            textFeatures.antonymScore > 0.4;

        // Split into sentences
        const sourceSentences = embeddingEngine.splitIntoSentences(sourceText);
        const hypSentences = embeddingEngine.splitIntoSentences(hypothesisText);

        // Get embeddings
        const embedStart = performance.now();
        const sourceEmbeddings = await embeddingEngine.embedBatch(sourceSentences);
        const hypEmbeddings = await embeddingEngine.embedBatch(hypSentences);
        this.metrics.embedTime = performance.now() - embedStart;

        let result;

        // If obvious factual hallucination detected, bypass cascade
        if (obviousHallucination) {
            this.metrics.stage = 'text-analysis';
            const energy = Math.max(textFeatures.numberMismatch, textFeatures.antonymScore);
            result = {
                energy: energy,
                isHallucination: true,
                verdict: 'FAIL',
                confidence: 0.9,
                stage: 'Text Analysis',
                stageReason: `Factual mismatch detected: ${textFeatures.numberMismatch > 0.3 ? 'numbers differ' : 'semantic contradiction'
                    }`,
                textFeatures
            };
        } else if (this.params.useCascade) {
            // STAGE 1: Feature-based ReGA
            const featureStart = performance.now();
            const featureResult = this.featureReGA(sourceEmbeddings, hypEmbeddings, textFeatures);
            this.metrics.featureTime = performance.now() - featureStart;

            if (!featureResult.isAmbiguous) {
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
                // STAGE 2: Deep ReGA
                const deepStart = performance.now();
                const deepResult = this.deepReGA(sourceEmbeddings, hypEmbeddings, textFeatures);
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
            const deepResult = this.deepReGA(sourceEmbeddings, hypEmbeddings, textFeatures);
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
                alignmentMatrix: result.deepResult?.permutation || this.computeSimpleAlignment(sourceEmbeddings, hypEmbeddings),
                textFeatures
            }
        };
    }

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
