/**
 * ReGA Engine - Graph-Based RAG Verification
 * 
 * Based on: "ReGA: Zero-Cost Graph Alignment for Structural Hallucination Detection"
 * 
 * Features:
 * - Proper energy calibration (identical texts = 0)
 * - White-box explainability (shows what triggered detection)
 * - Sinkhorn-based graph alignment
 */

import { embeddingEngine } from './embeddings.js';

class ReGAEngine {
    constructor() {
        this.params = {
            sinkhornIterations: 10,
            temperature: 1.0,
            matchSteps: 5,
            threshold: 0.15,
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
    // TEXT PREPROCESSING
    // =========================================================================

    normalizeText(text) {
        return text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    areTextsIdentical(source, hypothesis) {
        const normSource = this.normalizeText(source);
        const normHyp = this.normalizeText(hypothesis);

        if (normSource === normHyp) return true;

        const srcWords = new Set(normSource.split(' '));
        const hypWords = new Set(normHyp.split(' '));

        let overlap = 0;
        for (const word of srcWords) {
            if (hypWords.has(word)) overlap++;
        }

        const jaccardSim = overlap / (srcWords.size + hypWords.size - overlap);
        return jaccardSim > 0.95;
    }

    extractNumbers(text) {
        const numbers = [];
        const regex = /\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\b/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            numbers.push(parseFloat(match[0].replace(/,/g, '')));
        }
        return numbers;
    }

    extractEntities(text) {
        const entities = [];
        const regex = /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            entities.push(match[0].toLowerCase());
        }
        return [...new Set(entities)];
    }

    // =========================================================================
    // FACTUAL ANALYSIS WITH EXPLANATIONS
    // =========================================================================

    computeFactualFeatures(sourceText, hypText) {
        const explanations = [];

        // Number comparison
        const srcNums = this.extractNumbers(sourceText);
        const hypNums = this.extractNumbers(hypText);
        const mismatchedNumbers = [];

        let numberMismatch = 0;
        if (srcNums.length > 0 && hypNums.length > 0) {
            for (const hypNum of hypNums) {
                const matches = srcNums.some(srcNum =>
                    Math.abs(hypNum - srcNum) / Math.max(srcNum, 1) < 0.01
                );
                if (!matches) {
                    numberMismatch += 1;
                    const closest = srcNums.reduce((a, b) =>
                        Math.abs(b - hypNum) < Math.abs(a - hypNum) ? b : a
                    );
                    mismatchedNumbers.push({ hypothesis: hypNum, source: closest });
                }
            }
            numberMismatch = numberMismatch / hypNums.length;
        }

        if (mismatchedNumbers.length > 0) {
            for (const m of mismatchedNumbers) {
                explanations.push({
                    type: 'number',
                    icon: '🔢',
                    text: `Number mismatch: "${m.hypothesis}" should be "${m.source}"`
                });
            }
        }

        // Entity comparison
        const srcEntities = new Set(this.extractEntities(sourceText));
        const hypEntities = new Set(this.extractEntities(hypText));
        const newEntities = [];

        for (const entity of hypEntities) {
            if (!srcEntities.has(entity)) {
                newEntities.push(entity);
            }
        }

        let entityMismatch = 0;
        if (hypEntities.size > 0) {
            entityMismatch = newEntities.length / hypEntities.size;
        }

        if (newEntities.length > 0) {
            explanations.push({
                type: 'entity',
                icon: '👤',
                text: `Unknown entities: ${newEntities.slice(0, 3).map(e => `"${e}"`).join(', ')}${newEntities.length > 3 ? '...' : ''}`
            });
        }

        // Antonym detection
        const antonymPairs = [
            ['tallest', 'smallest'], ['largest', 'smallest'], ['highest', 'lowest'],
            ['best', 'worst'], ['first', 'last'], ['increase', 'decrease'],
            ['always', 'never'], ['all', 'none'], ['true', 'false'],
            ['bought', 'sold'], ['acquired', 'divested'], ['created', 'destroyed'],
            ['success', 'failure'], ['win', 'lose'], ['positive', 'negative'],
            ['approved', 'rejected'], ['accept', 'reject'], ['allow', 'deny'],
            ['started', 'ended'], ['opened', 'closed'], ['raised', 'lowered']
        ];

        const srcLower = sourceText.toLowerCase();
        const hypLower = hypText.toLowerCase();
        const foundAntonyms = [];

        for (const [word1, word2] of antonymPairs) {
            if (srcLower.includes(word1) && hypLower.includes(word2)) {
                foundAntonyms.push({ source: word1, hypothesis: word2 });
            } else if (srcLower.includes(word2) && hypLower.includes(word1)) {
                foundAntonyms.push({ source: word2, hypothesis: word1 });
            }
        }

        if (foundAntonyms.length > 0) {
            for (const a of foundAntonyms) {
                explanations.push({
                    type: 'antonym',
                    icon: '⚠️',
                    text: `Contradiction: "${a.source}" → "${a.hypothesis}"`
                });
            }
        }

        return {
            numberMismatch,
            entityMismatch,
            antonymFound: foundAntonyms.length > 0 ? 1 : 0,
            explanations,
            details: {
                mismatchedNumbers,
                newEntities,
                foundAntonyms
            }
        };
    }

    // =========================================================================
    // ALIGNMENT FEATURES
    // =========================================================================

    computeAlignmentFeatures(sourceEmbeddings, hypEmbeddings) {
        const n = sourceEmbeddings.length;
        const m = hypEmbeddings.length;

        const simMatrix = [];
        for (let i = 0; i < m; i++) {
            const row = [];
            for (let j = 0; j < n; j++) {
                row.push(this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            simMatrix.push(row);
        }

        const alignments = [];
        for (let i = 0; i < m; i++) {
            const maxSim = Math.max(...simMatrix[i]);
            const bestMatch = simMatrix[i].indexOf(maxSim);
            alignments.push({ hypIdx: i, srcIdx: bestMatch, similarity: maxSim });
        }

        const avgSimilarity = alignments.reduce((sum, a) => sum + a.similarity, 0) / alignments.length;
        const minSimilarity = Math.min(...alignments.map(a => a.similarity));

        return {
            avgCost: 1 - avgSimilarity,
            maxCost: 1 - minSimilarity,
            alignments,
            simMatrix
        };
    }

    // =========================================================================
    // DEEP REGA - SINKHORN ALIGNMENT
    // =========================================================================

    sinkhorn(matrix, iters = 10, temperature = 1.0) {
        const rows = matrix.length;
        const cols = matrix[0].length;

        let P = matrix.map(row =>
            row.map(val => Math.exp(val / temperature))
        );

        for (let iter = 0; iter < iters; iter++) {
            for (let i = 0; i < rows; i++) {
                const rowSum = P[i].reduce((a, b) => a + b, 0) + 1e-8;
                P[i] = P[i].map(v => v / rowSum);
            }
            for (let j = 0; j < cols; j++) {
                let colSum = 0;
                for (let i = 0; i < rows; i++) colSum += P[i][j];
                colSum += 1e-8;
                for (let i = 0; i < rows; i++) P[i][j] /= colSum;
            }
        }

        return P;
    }

    computeStructuralEnergy(simMatrix, permutation) {
        const m = permutation.length;
        const n = permutation[0].length;

        let energy = 0;
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                const cost = 1 - simMatrix[i][j];
                energy += permutation[i][j] * cost;
            }
        }

        return energy;
    }

    deepReGA(sourceEmbeddings, hypEmbeddings, explanations) {
        const simMatrix = [];
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const row = [];
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                row.push(this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            simMatrix.push(row);
        }

        const permutation = this.sinkhorn(
            simMatrix,
            this.params.sinkhornIterations,
            this.params.temperature
        );

        const energy = this.computeStructuralEnergy(simMatrix, permutation);

        // Add explanation for low-similarity alignments
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const maxSim = Math.max(...simMatrix[i]);
            if (maxSim < 0.7) {
                explanations.push({
                    type: 'alignment',
                    icon: '🔗',
                    text: `Weak alignment for sentence ${i + 1} (${(maxSim * 100).toFixed(0)}% similarity)`
                });
            }
        }

        return {
            energy,
            permutation,
            simMatrix
        };
    }

    // =========================================================================
    // MAIN VERIFICATION PIPELINE
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

        // Quick check: identical texts
        if (this.areTextsIdentical(sourceText, hypothesisText)) {
            this.metrics.totalTime = performance.now() - startTime;
            this.metrics.stage = 'identical';
            return {
                energy: 0,
                isHallucination: false,
                verdict: 'PASS',
                confidence: 1.0,
                stage: 'Identical Text',
                stageReason: 'Source and hypothesis are essentially identical',
                explanations: [],
                metrics: { ...this.metrics },
                details: {
                    sourceSentences: [sourceText],
                    hypSentences: [hypothesisText],
                    sourceEmbeddings: [],
                    hypEmbeddings: [],
                    alignmentMatrix: [[1]]
                }
            };
        }

        // Split into sentences
        const sourceSentences = embeddingEngine.splitIntoSentences(sourceText);
        const hypSentences = embeddingEngine.splitIntoSentences(hypothesisText);

        // Get embeddings
        const embedStart = performance.now();
        const sourceEmbeddings = await embeddingEngine.embedBatch(sourceSentences);
        const hypEmbeddings = await embeddingEngine.embedBatch(hypSentences);
        this.metrics.embedTime = performance.now() - embedStart;

        // Stage 1: Compute factual features with explanations
        const featureStart = performance.now();
        const factualFeatures = this.computeFactualFeatures(sourceText, hypothesisText);
        const explanations = [...factualFeatures.explanations];
        this.metrics.featureTime = performance.now() - featureStart;

        // Stage 2: Deep ReGA
        const deepStart = performance.now();
        const deepResult = this.deepReGA(sourceEmbeddings, hypEmbeddings, explanations);
        this.metrics.deepRegaTime = performance.now() - deepStart;

        // Combine energies
        let finalEnergy = deepResult.energy;

        if (factualFeatures.numberMismatch > 0) {
            finalEnergy += factualFeatures.numberMismatch * 0.3;
        }
        if (factualFeatures.antonymFound) {
            finalEnergy += 0.4;
        }
        if (factualFeatures.entityMismatch > 0.3) {
            finalEnergy += factualFeatures.entityMismatch * 0.2;
        }

        finalEnergy = Math.max(0, Math.min(1, finalEnergy));

        const isHallucination = finalEnergy > this.params.threshold;
        const confidence = isHallucination
            ? Math.min((finalEnergy - this.params.threshold) / 0.3 + 0.5, 1.0)
            : Math.min((this.params.threshold - finalEnergy) / this.params.threshold + 0.5, 1.0);

        let stage = 'Deep ReGA';
        let stageReason = 'Graph alignment with Sinkhorn normalization';

        if (factualFeatures.antonymFound) {
            stage = 'Factual Analysis';
            stageReason = 'Semantic contradiction detected';
        } else if (factualFeatures.numberMismatch > 0.2) {
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
                alignmentMatrix: deepResult.permutation,
                factualFeatures
            }
        };
    }

    // =========================================================================
    // UTILITY
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
}

export const regaEngine = new ReGAEngine();
export default regaEngine;
