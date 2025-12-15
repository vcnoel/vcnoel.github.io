/**
 * ReGA Demo Application
 * Main entry point that wires up all components
 */

import { embeddingEngine } from './embeddings.js';
import { regaEngine } from './rega-engine.js';
import { graphVisualizer } from './visualizer.js';

// ============================================================================
// Preset Examples
// ============================================================================

const PRESETS = {
    business: {
        source: "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California. Tim Cook became CEO in August 2011, succeeding Steve Jobs. Apple's market capitalization exceeded $3 trillion in 2022.",
        hypothesis: "Apple was founded by Steve Jobs and Bill Gates in 1975. The company is based in San Francisco, California. Steve Jobs is currently serving as the CEO. Apple became the first company to reach $1 trillion market cap."
    },
    medical: {
        source: "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency. Risk factors include obesity, sedentary lifestyle, and genetic predisposition. Treatment typically involves lifestyle modifications, oral medications like metformin, and sometimes insulin therapy. Regular monitoring of blood glucose levels is essential.",
        hypothesis: "Type 2 diabetes is caused by excessive sugar intake alone. The only treatment is insulin injections. Patients with Type 2 diabetes always need insulin from diagnosis. Blood glucose monitoring is optional and only needed during illness."
    },
    legal: {
        source: "The Fourth Amendment to the United States Constitution protects citizens against unreasonable searches and seizures. A warrant is generally required for searches, which must be issued by a judge and supported by probable cause. Exceptions include consent searches, plain view doctrine, and exigent circumstances.",
        hypothesis: "The Fourth Amendment allows police to search any property without restrictions. Warrants are never required for law enforcement searches. Judges cannot issue search warrants. There are no exceptions to warrant requirements."
    },
    academic: {
        source: "The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al. It relies entirely on self-attention mechanisms, eliminating the need for recurrence. Key innovations include multi-head attention and positional encoding. The architecture has become foundational for modern NLP models like BERT and GPT.",
        hypothesis: "The transformer was invented by OpenAI in 2019. It uses recurrent neural networks as its core mechanism. The architecture does not use attention mechanisms. Transformers were first used for computer vision, not NLP."
    },
    faithful: {
        source: "The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair. It was designed by Gustave Eiffel's engineering company. The tower stands 330 meters tall and was the world's tallest man-made structure until 1930.",
        hypothesis: "The Eiffel Tower was built between 1887 and 1889 for the World's Fair. Gustave Eiffel's company designed it. It is 330 meters tall and held the record for the world's tallest structure until 1930."
    },
    directional: {
        source: "Microsoft acquired Activision Blizzard in January 2022 for $68.7 billion. This made Microsoft the third-largest gaming company by revenue. The acquisition was completed after regulatory approval.",
        hypothesis: "Activision Blizzard acquired Microsoft in January 2022 for $68.7 billion. This made Activision the third-largest gaming company by revenue. The acquisition was completed after regulatory approval."
    }
};

// ============================================================================
// Batch Test Data Generator
// ============================================================================

function generateBatchData(count) {
    const pairs = [];
    const templates = [
        {
            source: "Company X was founded in {year} by {founder}. It is headquartered in {city}.",
            hallucinated: "Company X was founded in {wrongYear} by {wrongFounder}. It is based in {wrongCity}.",
            faithful: "Company X was established in {year} by {founder}. Its headquarters is in {city}."
        },
        {
            source: "The study found that treatment A reduced symptoms by {percent}% in {weeks} weeks.",
            hallucinated: "The study showed treatment A eliminated all symptoms immediately with {wrongPercent}% success.",
            faithful: "Research indicated treatment A decreased symptoms by {percent}% over {weeks} weeks."
        },
        {
            source: "Product Y costs ${price} and includes {features} features. It was released in {year}.",
            hallucinated: "Product Y is free and has unlimited features. It has been available since {wrongYear}.",
            faithful: "Product Y is priced at ${price} with {features} features, launched in {year}."
        }
    ];

    const values = {
        year: [2020, 2019, 2018, 2021, 2022],
        wrongYear: [1990, 1985, 2030, 2025, 2000],
        founder: ['John Smith', 'Jane Doe', 'Alex Johnson', 'Maria Garcia'],
        wrongFounder: ['Unknown Person', 'Different CEO', 'Wrong Name', 'Fictional Founder'],
        city: ['New York', 'London', 'Tokyo', 'Paris', 'Berlin'],
        wrongCity: ['Mars City', 'Atlantis', 'Wrong Location', 'Fictional Place'],
        percent: [25, 40, 55, 70, 85],
        wrongPercent: [100, 150, 200, 99],
        weeks: [4, 6, 8, 12],
        price: [99, 199, 299, 499],
        features: [5, 10, 15, 20]
    };

    for (let i = 0; i < count; i++) {
        const template = templates[i % templates.length];
        const isHallucination = Math.random() > 0.5;

        let source = template.source;
        let hypothesis = isHallucination ? template.hallucinated : template.faithful;

        // Fill in values
        for (const [key, vals] of Object.entries(values)) {
            const val = vals[Math.floor(Math.random() * vals.length)];
            source = source.replace(`{${key}}`, val);
            hypothesis = hypothesis.replace(`{${key}}`, val);
        }

        pairs.push({ source, hypothesis, label: isHallucination ? 1 : 0 });
    }

    return pairs;
}

// ============================================================================
// UI Controllers
// ============================================================================

class DemoApp {
    constructor() {
        this.isModelReady = false;
        this.isProcessing = false;
    }

    async initialize() {
        // Initialize UI elements
        this.bindElements();
        this.bindEvents();

        // Initialize visualizer
        graphVisualizer.initialize();

        // Initialize background animation
        this.initNetworkBackground();

        // Load embedding model
        await this.loadModel();
    }

    bindElements() {
        this.elements = {
            // Model status
            modelStatusCard: document.getElementById('modelStatusCard'),
            modelStatusText: document.getElementById('modelStatusText'),
            modelProgress: document.getElementById('modelProgress'),

            // Inputs
            sourceText: document.getElementById('sourceText'),
            hypothesisText: document.getElementById('hypothesisText'),
            verifyBtn: document.getElementById('verifyBtn'),

            // Results
            gaugeFill: document.getElementById('gaugeFill'),
            energyValue: document.getElementById('energyValue'),
            verdict: document.getElementById('verdict'),
            embedTime: document.getElementById('embedTime'),
            graphTime: document.getElementById('graphTime'),
            sinkhornTime: document.getElementById('sinkhornTime'),
            totalTime: document.getElementById('totalTime'),

            // Hyperparameters
            sinkhornIters: document.getElementById('sinkhornIters'),
            sinkhornItersValue: document.getElementById('sinkhornItersValue'),
            temperature: document.getElementById('temperature'),
            temperatureValue: document.getElementById('temperatureValue'),
            matchSteps: document.getElementById('matchSteps'),
            matchStepsValue: document.getElementById('matchStepsValue'),
            threshold: document.getElementById('threshold'),
            thresholdValue: document.getElementById('thresholdValue'),

            // Batch
            batchSize: document.getElementById('batchSize'),
            runBatchBtn: document.getElementById('runBatchBtn'),
            batchProcessed: document.getElementById('batchProcessed'),
            batchThroughput: document.getElementById('batchThroughput'),
            batchLatency: document.getElementById('batchLatency'),
            batchAccuracy: document.getElementById('batchAccuracy'),
            batchProgressBar: document.getElementById('batchProgressBar')
        };
    }

    bindEvents() {
        // Verify button
        this.elements.verifyBtn.addEventListener('click', () => this.runVerification());

        // Hyperparameter sliders
        this.elements.sinkhornIters.addEventListener('input', (e) => {
            this.elements.sinkhornItersValue.textContent = e.target.value;
            regaEngine.setParams({ sinkhornIterations: parseInt(e.target.value) });
        });

        this.elements.temperature.addEventListener('input', (e) => {
            this.elements.temperatureValue.textContent = parseFloat(e.target.value).toFixed(1);
            regaEngine.setParams({ temperature: parseFloat(e.target.value) });
        });

        this.elements.matchSteps.addEventListener('input', (e) => {
            this.elements.matchStepsValue.textContent = e.target.value;
            regaEngine.setParams({ matchSteps: parseInt(e.target.value) });
        });

        this.elements.threshold.addEventListener('input', (e) => {
            this.elements.thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
            regaEngine.setParams({ energyThreshold: parseFloat(e.target.value) });
        });

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => this.loadPreset(btn.dataset.preset));
        });

        // Batch demo
        this.elements.runBatchBtn.addEventListener('click', () => this.runBatchDemo());

        // Window resize
        window.addEventListener('resize', () => {
            graphVisualizer.resize();
            this.resizeNetworkBackground();
        });
    }

    async loadModel() {
        const success = await embeddingEngine.initialize({
            onProgress: (message, percentage) => {
                this.elements.modelStatusText.textContent = message;
                this.elements.modelProgress.style.width = `${percentage}%`;
            },
            onReady: () => {
                this.isModelReady = true;
                this.elements.modelStatusCard.querySelector('.status-icon').className = 'status-icon ready';
                this.elements.modelStatusCard.querySelector('.status-icon').innerHTML = '<i class="fas fa-check"></i>';
                this.elements.modelStatusText.textContent = 'Model ready! You can now verify texts.';
                this.elements.verifyBtn.disabled = false;
                this.elements.runBatchBtn.disabled = false;
            },
            onError: (error) => {
                this.elements.modelStatusCard.querySelector('.status-icon').className = 'status-icon error';
                this.elements.modelStatusCard.querySelector('.status-icon').innerHTML = '<i class="fas fa-times"></i>';
                this.elements.modelStatusText.textContent = `Error: ${error.message}`;
            }
        });

        return success;
    }

    async runVerification() {
        if (!this.isModelReady || this.isProcessing) return;

        const source = this.elements.sourceText.value.trim();
        const hypothesis = this.elements.hypothesisText.value.trim();

        if (!source || !hypothesis) {
            alert('Please enter both source and hypothesis texts.');
            return;
        }

        this.isProcessing = true;
        this.elements.verifyBtn.disabled = true;
        this.elements.verifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

        try {
            const result = await regaEngine.verify(source, hypothesis);
            this.displayResults(result);
            graphVisualizer.render(result.details);
        } catch (error) {
            console.error('Verification error:', error);
            alert('Error during verification: ' + error.message);
        } finally {
            this.isProcessing = false;
            this.elements.verifyBtn.disabled = false;
            this.elements.verifyBtn.innerHTML = '<i class="fas fa-search"></i> Verify Hypothesis';
        }
    }

    displayResults(result) {
        // Update energy gauge
        const energyPercent = Math.min(result.energy * 100, 100);
        const gaugeOffset = 251.2 * (1 - energyPercent / 100);
        this.elements.gaugeFill.style.strokeDashoffset = gaugeOffset;
        this.elements.energyValue.textContent = result.energy.toFixed(3);

        // Update verdict with stage info
        this.elements.verdict.className = `verdict ${result.isHallucination ? 'fail' : 'pass'}`;
        const stageLabel = result.stage || 'ReGA';
        this.elements.verdict.innerHTML = `
            <span class="verdict-text">${result.verdict}: ${result.isHallucination ? 'Hallucination Detected' : 'Text Appears Faithful'}</span>
            <span class="verdict-stage">${stageLabel}</span>
        `;

        // Update metrics based on cascade stage
        this.elements.embedTime.textContent = `${result.metrics.embedTime.toFixed(1)} ms`;

        if (result.metrics.featureTime !== undefined) {
            this.elements.graphTime.textContent = `${result.metrics.featureTime.toFixed(1)} ms`;
        } else {
            this.elements.graphTime.textContent = `-- ms`;
        }

        if (result.metrics.deepRegaTime !== undefined && result.metrics.deepRegaTime > 0) {
            this.elements.sinkhornTime.textContent = `${result.metrics.deepRegaTime.toFixed(1)} ms`;
        } else {
            this.elements.sinkhornTime.textContent = `-- ms`;
        }

        this.elements.totalTime.textContent = `${result.metrics.totalTime.toFixed(1)} ms`;
    }

    loadPreset(presetName) {
        const preset = PRESETS[presetName];
        if (!preset) return;

        this.elements.sourceText.value = preset.source;
        this.elements.hypothesisText.value = preset.hypothesis;

        // Auto-run verification
        if (this.isModelReady) {
            this.runVerification();
        }
    }

    async runBatchDemo() {
        if (!this.isModelReady || this.isProcessing) return;

        const batchSize = parseInt(this.elements.batchSize.value);
        const testData = generateBatchData(batchSize);

        this.isProcessing = true;
        this.elements.runBatchBtn.disabled = true;
        this.elements.runBatchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';

        const startTime = performance.now();
        let correct = 0;
        let totalLatency = 0;

        try {
            for (let i = 0; i < testData.length; i++) {
                const pair = testData[i];
                const result = await regaEngine.verify(pair.source, pair.hypothesis);

                const predicted = result.isHallucination ? 1 : 0;
                if (predicted === pair.label) correct++;
                totalLatency += result.metrics.totalTime;

                // Update progress
                const progress = ((i + 1) / testData.length) * 100;
                this.elements.batchProgressBar.style.width = `${progress}%`;
                this.elements.batchProcessed.textContent = `${i + 1} / ${testData.length}`;
            }

            const totalTime = performance.now() - startTime;
            const throughput = (testData.length / totalTime) * 1000;
            const avgLatency = totalLatency / testData.length;
            const accuracy = (correct / testData.length) * 100;

            this.elements.batchThroughput.textContent = `${throughput.toFixed(2)} pairs/sec`;
            this.elements.batchLatency.textContent = `${avgLatency.toFixed(1)} ms`;
            this.elements.batchAccuracy.textContent = `${accuracy.toFixed(1)}%`;

        } catch (error) {
            console.error('Batch demo error:', error);
            alert('Error during batch demo: ' + error.message);
        } finally {
            this.isProcessing = false;
            this.elements.runBatchBtn.disabled = false;
            this.elements.runBatchBtn.innerHTML = '<i class="fas fa-play"></i> Run Batch Demo';
        }
    }

    // Network background animation
    initNetworkBackground() {
        const canvas = document.getElementById('networkCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.networkCanvas = canvas;
        this.networkCtx = ctx;
        this.networkNodes = [];

        this.resizeNetworkBackground();

        // Create nodes
        for (let i = 0; i < 50; i++) {
            this.networkNodes.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1
            });
        }

        this.animateNetwork();
    }

    resizeNetworkBackground() {
        if (!this.networkCanvas) return;
        this.networkCanvas.width = window.innerWidth;
        this.networkCanvas.height = window.innerHeight;
    }

    animateNetwork() {
        if (!this.networkCtx) return;

        const ctx = this.networkCtx;
        const canvas = this.networkCanvas;
        const nodes = this.networkNodes;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Update and draw nodes
        for (const node of nodes) {
            node.x += node.vx;
            node.y += node.vy;

            // Bounce off edges
            if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
            if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

            // Draw node
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(99, 102, 241, 0.5)';
            ctx.fill();
        }

        // Draw connections
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${0.2 * (1 - dist / 150)})`;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(() => this.animateNetwork());
    }
}

// Initialize app when DOM is ready
const app = new DemoApp();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => app.initialize());
} else {
    app.initialize();
}

export default app;
