/**
 * ReGA Engine - Proper Implementation
 * 
 * Based on: "ReGA: Zero-Cost Graph Alignment for Structural Hallucination Detection"
 * 
 * Key principle: Energy should be ~0 for identical/faithful texts, high for hallucinations
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

    /**
     * Normalize text for comparison
     */
    normalizeText(text) {
        return text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    /**
     * Check if texts are essentially identical
     */
    areTextsIdentical(source, hypothesis) {
        const normSource = this.normalizeText(source);
        const normHyp = this.normalizeText(hypothesis);

        if (normSource === normHyp) return true;

        // Check similarity of normalized versions
        const srcWords = new Set(normSource.split(' '));
        const hypWords = new Set(normHyp.split(' '));

        let overlap = 0;
        for (const word of srcWords) {
            if (hypWords.has(word)) overlap++;
        }

        const jaccardSim = overlap / (srcWords.size + hypWords.size - overlap);
        return jaccardSim > 0.95;  // 95% word overlap = essentially identical
    }

    /**
     * Extract numbers from text
     */
    extractNumbers(text) {
        const numbers = [];
        const regex = /\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?\b/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            numbers.push(parseFloat(match[0].replace(/,/g, '')));
        }
        return numbers;
    }

    /**
     * Extract key entities (proper nouns, capitalized phrases)
     */
    extractEntities(text) {
        const entities = [];
        // Match capitalized words and multi-word proper nouns
        const regex = /[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g;
        let match;
        while ((match = regex.exec(text)) !== null) {
            entities.push(match[0].toLowerCase());
        }
        return [...new Set(entities)];
    }

    // =========================================================================
    // FEATURE-BASED REGA (Stage 1)
    // =========================================================================

    /**
     * Compute alignment cost between source and hypothesis embeddings
     * This is the core of Feature ReGA
     */
    computeAlignmentFeatures(sourceEmbeddings, hypEmbeddings) {
        const n = sourceEmbeddings.length;
        const m = hypEmbeddings.length;

        // Build similarity matrix
        const simMatrix = [];
        for (let i = 0; i < m; i++) {
            const row = [];
            for (let j = 0; j < n; j++) {
                row.push(this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            simMatrix.push(row);
        }

        // For each hypothesis sentence, find best matching source sentence
        const alignments = [];
        for (let i = 0; i < m; i++) {
            const maxSim = Math.max(...simMatrix[i]);
            const bestMatch = simMatrix[i].indexOf(maxSim);
            alignments.push({ hypIdx: i, srcIdx: bestMatch, similarity: maxSim });
        }

        // Compute alignment quality score
        const avgSimilarity = alignments.reduce((sum, a) => sum + a.similarity, 0) / alignments.length;
        const minSimilarity = Math.min(...alignments.map(a => a.similarity));

        // Cost = 1 - similarity (low similarity = high cost = likely hallucination)
        return {
            avgCost: 1 - avgSimilarity,
            maxCost: 1 - minSimilarity,
            alignments,
            simMatrix
        };
    }

    /**
     * Compute factual consistency features
     */
    computeFactualFeatures(sourceText, hypText) {
        // Number comparison
        const srcNums = this.extractNumbers(sourceText);
        const hypNums = this.extractNumbers(hypText);

        let numberMismatch = 0;
        if (srcNums.length > 0 && hypNums.length > 0) {
            // Check if hypothesis numbers match source numbers
            for (const hypNum of hypNums) {
                const matches = srcNums.some(srcNum =>
                    Math.abs(hypNum - srcNum) / Math.max(srcNum, 1) < 0.01  // 1% tolerance
                );
                if (!matches) {
                    numberMismatch += 1;
                }
            }
            numberMismatch = numberMismatch / hypNums.length;
        }

        // Entity comparison
        const srcEntities = new Set(this.extractEntities(sourceText));
        const hypEntities = new Set(this.extractEntities(hypText));

        let entityMismatch = 0;
        if (hypEntities.size > 0) {
            let missing = 0;
            for (const entity of hypEntities) {
                if (!srcEntities.has(entity)) missing++;
            }
            entityMismatch = missing / hypEntities.size;
        }

        // Antonym detection
        const antonyms = [
            ['tallest', 'smallest'], ['largest', 'smallest'], ['highest', 'lowest'],
            ['best', 'worst'], ['first', 'last'], ['increase', 'decrease'],
            ['always', 'never'], ['all', 'none'], ['true', 'false'],
            ['bought', 'sold'], ['acquired', 'divested'], ['created', 'destroyed']
        ];

        const srcLower = sourceText.toLowerCase();
        const hypLower = hypText.toLowerCase();
        let antonymFound = false;
        for (const [word1, word2] of antonyms) {
            if ((srcLower.includes(word1) && hypLower.includes(word2)) ||
                (srcLower.includes(word2) && hypLower.includes(word1))) {
                antonymFound = true;
                break;
            }
        }

        return {
            numberMismatch,
            entityMismatch,
            antonymFound: antonymFound ? 1 : 0
        };
    }

    // =========================================================================
    // DEEP REGA (Stage 2) - Graph-based alignment
    // =========================================================================

    /**
     * Sinkhorn-Knopp normalization
     */
    sinkhorn(matrix, iters = 10, temperature = 1.0) {
        const rows = matrix.length;
        const cols = matrix[0].length;

        // Scale by temperature
        let P = matrix.map(row =>
            row.map(val => Math.exp(val / temperature))
        );

        // Normalize
        for (let iter = 0; iter < iters; iter++) {
            // Row normalization
            for (let i = 0; i < rows; i++) {
                const rowSum = P[i].reduce((a, b) => a + b, 0) + 1e-8;
                P[i] = P[i].map(v => v / rowSum);
            }
            // Column normalization
            for (let j = 0; j < cols; j++) {
                let colSum = 0;
                for (let i = 0; i < rows; i++) colSum += P[i][j];
                colSum += 1e-8;
                for (let i = 0; i < rows; i++) P[i][j] /= colSum;
            }
        }

        return P;
    }

    /**
     * Compute structural energy from alignment
     * Energy = how well the hypothesis aligns with source
     * Low energy = good alignment = faithful
     * High energy = poor alignment = hallucination
     */
    computeStructuralEnergy(simMatrix, permutation) {
        const m = permutation.length;
        const n = permutation[0].length;

        // Compute weighted alignment cost
        let energy = 0;
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                // Cost is (1 - similarity), weighted by alignment probability
                const cost = 1 - simMatrix[i][j];
                energy += permutation[i][j] * cost;
            }
        }

        return energy;
    }

    /**
     * Deep ReGA with graph neural network style processing
     */
    deepReGA(sourceEmbeddings, hypEmbeddings) {
        // Build similarity matrix
        const simMatrix = [];
        for (let i = 0; i < hypEmbeddings.length; i++) {
            const row = [];
            for (let j = 0; j < sourceEmbeddings.length; j++) {
                row.push(this.cosineSimilarity(hypEmbeddings[i], sourceEmbeddings[j]));
            }
            simMatrix.push(row);
        }

        // Apply Sinkhorn to get soft permutation
        const permutation = this.sinkhorn(
            simMatrix,
            this.params.sinkhornIterations,
            this.params.temperature
        );

        // Compute structural energy
        const energy = this.computeStructuralEnergy(simMatrix, permutation);

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

        // Stage 1: Compute factual features
        const featureStart = performance.now();
        const factualFeatures = this.computeFactualFeatures(sourceText, hypothesisText);
        const alignmentFeatures = this.computeAlignmentFeatures(sourceEmbeddings, hypEmbeddings);
        this.metrics.featureTime = performance.now() - featureStart;

        // Stage 2: Deep ReGA
        const deepStart = performance.now();
        const deepResult = this.deepReGA(sourceEmbeddings, hypEmbeddings);
        this.metrics.deepRegaTime = performance.now() - deepStart;

        // Combine energies with weights
        // - Alignment energy: base semantic alignment
        // - Factual penalties: for specific factual errors
        let finalEnergy = deepResult.energy;

        // Add penalties for factual mismatches
        if (factualFeatures.numberMismatch > 0) {
            finalEnergy += factualFeatures.numberMismatch * 0.3;
        }
        if (factualFeatures.antonymFound) {
            finalEnergy += 0.4;  // Major penalty for contradictions
        }
        if (factualFeatures.entityMismatch > 0.3) {
            finalEnergy += factualFeatures.entityMismatch * 0.2;
        }

        // Clamp energy to [0, 1]
        finalEnergy = Math.max(0, Math.min(1, finalEnergy));

        // Make decision
        const isHallucination = finalEnergy > this.params.threshold;
        const confidence = isHallucination
            ? Math.min((finalEnergy - this.params.threshold) / 0.3 + 0.5, 1.0)
            : Math.min((this.params.threshold - finalEnergy) / this.params.threshold + 0.5, 1.0);

        // Determine which stage contributed most
        let stage = 'Deep ReGA';
        let stageReason = 'Graph alignment with Sinkhorn normalization';

        if (factualFeatures.antonymFound) {
            stage = 'Factual Analysis';
            stageReason = 'Semantic contradiction detected (antonyms)';
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
            metrics: { ...this.metrics },
            details: {
                sourceSentences,
                hypSentences,
                sourceEmbeddings: sourceEmbeddings.map(e => Array.from(e)),
                hypEmbeddings: hypEmbeddings.map(e => Array.from(e)),
                alignmentMatrix: deepResult.permutation,
                factualFeatures,
                alignmentFeatures,
                deepResult
            }
        };
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
}

export const regaEngine = new ReGAEngine();
export default regaEngine;
