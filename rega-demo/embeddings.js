/**
 * Embeddings Module - Transformers.js Wrapper
 * Uses all-MiniLM-L6-v2 for in-browser sentence embeddings
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// Configure Transformers.js to use CDN for models
env.allowLocalModels = false;
env.useBrowserCache = true;

class EmbeddingEngine {
    constructor() {
        this.embedder = null;
        this.modelName = 'Xenova/all-MiniLM-L6-v2';
        this.isLoading = false;
        this.isReady = false;
        this.callbacks = {
            onProgress: null,
            onReady: null,
            onError: null
        };
    }

    /**
     * Initialize the embedding model
     */
    async initialize(callbacks = {}) {
        if (this.isReady) return true;
        if (this.isLoading) return false;

        this.callbacks = { ...this.callbacks, ...callbacks };
        this.isLoading = true;

        try {
            this.updateProgress('Loading embedding model...', 10);

            this.embedder = await pipeline('feature-extraction', this.modelName, {
                progress_callback: (progress) => {
                    if (progress.status === 'progress') {
                        const pct = Math.round((progress.loaded / progress.total) * 80) + 10;
                        this.updateProgress(`Downloading ${progress.file}...`, pct);
                    } else if (progress.status === 'done') {
                        this.updateProgress('Model loaded!', 90);
                    }
                }
            });

            this.isReady = true;
            this.isLoading = false;
            this.updateProgress('Ready!', 100);

            if (this.callbacks.onReady) {
                this.callbacks.onReady();
            }

            return true;
        } catch (error) {
            this.isLoading = false;
            console.error('Failed to load embedding model:', error);
            
            if (this.callbacks.onError) {
                this.callbacks.onError(error);
            }
            
            return false;
        }
    }

    updateProgress(message, percentage) {
        if (this.callbacks.onProgress) {
            this.callbacks.onProgress(message, percentage);
        }
    }

    /**
     * Get embedding for a single text
     * @param {string} text - Text to embed
     * @returns {Float32Array} - Embedding vector (384 dimensions)
     */
    async embed(text) {
        if (!this.isReady) {
            throw new Error('Embedding model not ready');
        }

        const output = await this.embedder(text, {
            pooling: 'mean',
            normalize: true
        });

        return output.data;
    }

    /**
     * Get embeddings for multiple texts
     * @param {string[]} texts - Array of texts
     * @returns {Float32Array[]} - Array of embedding vectors
     */
    async embedBatch(texts) {
        if (!this.isReady) {
            throw new Error('Embedding model not ready');
        }

        const embeddings = [];
        for (const text of texts) {
            const embedding = await this.embed(text);
            embeddings.push(embedding);
        }

        return embeddings;
    }

    /**
     * Split text into sentences
     * @param {string} text - Text to split
     * @returns {string[]} - Array of sentences
     */
    splitIntoSentences(text) {
        // Split by sentence boundaries
        const sentences = text
            .split(/(?<=[.!?])\s+/)
            .map(s => s.trim())
            .filter(s => s.length > 0);
        
        // If no sentences found, return the whole text
        if (sentences.length === 0) {
            return [text];
        }

        return sentences;
    }

    /**
     * Get embedding dimension
     */
    getDimension() {
        return 384; // MiniLM-L6-v2 produces 384-dimensional embeddings
    }
}

// Export singleton instance
export const embeddingEngine = new EmbeddingEngine();
export default embeddingEngine;
