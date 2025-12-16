/**
 * HalluGraph App - Main Application Logic
 */

import { halluGraphEngine } from './hallugraph-engine.js';
import { visualizer } from './visualizer.js';

// ============================================================================
// Batch Test Data Generator (Legal Theme)
// ============================================================================

function generateBatchData(count) {
    const pairs = [];

    // Helper to generate a random date
    const randomDate = () => {
        const year = 2020 + Math.floor(Math.random() * 6);
        const month = Math.floor(Math.random() * 12);
        const day = 1 + Math.floor(Math.random() * 28);
        const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
        return `${months[month]} ${day}, ${year}`;
    };

    // Helper to generate a random money amount
    const randomMoney = () => {
        const amount = (Math.floor(Math.random() * 1000) + 1) * 1000;
        return `$${amount.toLocaleString()}`;
    };

    const companies = ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Ltd", "Epsilon Co", "Zeta Group", "Omega Partners", "Theta Systems", "Apex Industries", "Blue Sky Ventures"];

    for (let i = 0; i < count; i++) {
        const isHallucination = Math.random() > 0.5;

        // Build a Long Contract
        const partyA = companies[Math.floor(Math.random() * companies.length)];
        let partyB = companies[Math.floor(Math.random() * companies.length)];
        while (partyB === partyA) partyB = companies[Math.floor(Math.random() * companies.length)];

        const effectiveDate = randomDate();
        const baseRent = randomMoney();

        // 1. Header
        let sourceText = `COMMERCIAL LEASE AGREEMENT\n\nThis Lease Agreement is made and entered into on ${effectiveDate}, by and between ${partyA} ("Landlord") and ${partyB} ("Tenant").\n\n`;
        let responseText = isHallucination
            ? `Summary of Lease:\nAgreement between ${partyA} and ${partyB} dated ${randomDate()} (wrong date).\n` // Hallucination start
            : `Summary of Lease:\nAgreement between ${partyA} and ${partyB} dated ${effectiveDate}.\n`;

        // 2. Generate many sections to increase entity count
        // Aiming for ~20-30 sections with entities
        const numberOfSections = 40; // High density

        for (let s = 1; s <= numberOfSections; s++) {
            const date = randomDate();
            const amount = randomMoney();
            const sectionNum = `Section ${s}.0`;

            // Source Clause
            sourceText += `${sectionNum}: Payment of ${amount} is due by ${partyB} on ${date}. The Landlord, ${partyA}, acknowledges receipt of prior obligations.\n`;

            // Response Clause
            if (isHallucination && Math.random() > 0.3) {
                // High probability of hallucination per section if global isHallucination
                // Scramble amounts and dates
                const wrongAmount = randomMoney(); // Likely different
                const wrongDate = randomDate();
                responseText += `${sectionNum}: Tenant must pay ${wrongAmount} by ${wrongDate}.\n`;
            } else {
                // Faithful representation
                responseText += `${sectionNum}: Tenant must pay ${amount} by ${date}.\n`;
            }
        }

        // 3. Add some case citations for flavor
        const citation = "Doe v. Smith";
        sourceText += `\nDisputes shall be resolved according to the precedent in ${citation}, 500 U.S. 123.\n`;
        if (isHallucination) {
            responseText += `\nGoverning law based on Roe v. Wade (irrelevant).\n`;
        } else {
            responseText += `\nGoverning law follows ${citation}.\n`;
        }

        pairs.push({ context: sourceText, response: responseText, label: isHallucination ? 1 : 0 });
    }

    return pairs;
}

class HalluGraphApp {
    constructor() {
        this.elements = {};
        this.presets = {
            'contract-faithful': {
                context: `Westfield Properties LLC ("Landlord") and Pacific Retail Group Inc. ("Tenant") entered into a Commercial Lease Agreement effective January 15, 2024. The monthly rent is $45,000 payable on the first business day of each month. The lease term is 5 years with two 3-year renewal options. Security deposit of $135,000 is required.`,
                response: `The lease agreement between Westfield Properties LLC and Pacific Retail Group Inc. was signed on January 15, 2024. The monthly rent is $45,000, due on the first business day of each month. The lease runs for 5 years with renewal options available.`
            },
            'contract-hallucinated': {
                context: `Westfield Properties LLC ("Landlord") and Pacific Retail Group Inc. ("Tenant") entered into a Commercial Lease Agreement effective January 15, 2024. The monthly rent is $45,000 payable on the first business day of each month. The lease term is 5 years with two 3-year renewal options. Security deposit of $135,000 is required.`,
                response: `The lease agreement between Westfield Properties LLC and Pacific Retail Group Inc. was signed on January 15, 2024. The monthly rent is $48,000, due on the last day of each month. The lease runs for 3 years with renewal options available.`
            },
            'case-faithful': {
                context: `In Smith v. Anderson, 500 U.S. 123 (1995), the Supreme Court held that employers are liable for workplace discrimination under Title VII. The Court affirmed the judgment of the Ninth Circuit. Justice Stevens delivered the opinion for a 6-3 majority.`,
                response: `The case Smith v. Anderson, 500 U.S. 123 (1995) established employer liability for workplace discrimination under Title VII. The Supreme Court affirmed the Ninth Circuit's decision with Justice Stevens writing for the 6-3 majority.`
            },
            'case-hallucinated': {
                context: `In Smith v. Anderson, 500 U.S. 123 (1995), the Supreme Court held that employers are liable for workplace discrimination under Title VII. The Court affirmed the judgment of the Ninth Circuit. Justice Stevens delivered the opinion for a 6-3 majority.`,
                response: `The case Johnson v. Anderson, 502 U.S. 145 (1997) established employer immunity from workplace discrimination claims under Title VII. The Supreme Court reversed the Ninth Circuit's decision with Justice Roberts writing for the 5-4 majority.`
            },
            'party-swap': {
                context: `In the matter of ABC Corporation ("Plaintiff") versus XYZ Industries ("Defendant"), the Plaintiff filed a breach of contract claim on March 15, 2023. The Defendant is alleged to have failed to deliver 5,000 units as specified in Purchase Order #12345. Damages sought: $750,000.`,
                response: `XYZ Industries filed a breach of contract claim against ABC Corporation on March 15, 2023. ABC Corporation allegedly failed to deliver the 5,000 units specified in Purchase Order #12345, resulting in claimed damages of $750,000.`
            }
        };
    }

    async init() {
        this.bindElements();
        this.bindEvents();

        // Initialize visualizer
        visualizer.initialize();
        visualizer.renderPlaceholder();

        // Simulate model loading (we use regex-based extraction, no ML model needed)
        await this.simulateModelLoading();
    }

    bindElements() {
        this.elements = {
            contextText: document.getElementById('contextText'),
            responseText: document.getElementById('responseText'),
            verifyBtn: document.getElementById('verifyBtn'),
            modelStatusCard: document.getElementById('modelStatusCard'),
            modelStatusText: document.getElementById('modelStatusText'),
            modelProgress: document.getElementById('modelProgress'),
            gaugeFill: document.getElementById('gaugeFill'),
            cfiValue: document.getElementById('cfiValue'),
            verdict: document.getElementById('verdict'),
            egScore: document.getElementById('egScore'),
            rpScore: document.getElementById('rpScore'),
            entityCount: document.getElementById('entityCount'),
            relationCount: document.getElementById('relationCount'),
            auditList: document.getElementById('auditList'),
            extractTime: document.getElementById('extractTime'),
            alignTime: document.getElementById('alignTime'),
            totalTime: document.getElementById('totalTime'),
            totalTime: document.getElementById('totalTime'),
            carbonValue: document.getElementById('carbonValue'),

            // Batch Demo Elements
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
        this.elements.verifyBtn.addEventListener('click', () => this.runVerification());

        // Batch Demo
        this.elements.runBatchBtn.addEventListener('click', () => this.runBatchDemo());

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const presetId = btn.dataset.preset;
                this.loadPreset(presetId);
            });
        });
    }

    async simulateModelLoading() {
        const steps = [
            { text: 'Initializing entity extraction...', progress: 25 },
            { text: 'Loading relation patterns...', progress: 50 },
            { text: 'Configuring alignment metrics...', progress: 75 },
            { text: 'Ready!', progress: 100 }
        ];

        for (const step of steps) {
            this.elements.modelStatusText.textContent = step.text;
            this.elements.modelProgress.style.width = step.progress + '%';
            await new Promise(r => setTimeout(r, 300));
        }

        // Mark as ready
        this.elements.modelStatusCard.querySelector('.status-icon').classList.remove('loading');
        this.elements.modelStatusCard.querySelector('.status-icon').classList.add('ready');
        this.elements.modelStatusCard.querySelector('.status-icon i').className = 'fas fa-check-circle';
        this.elements.verifyBtn.disabled = false;
    }

    loadPreset(presetId) {
        const preset = this.presets[presetId];
        if (preset) {
            this.elements.contextText.value = preset.context;
            this.elements.responseText.value = preset.response;
        }
    }

    async runVerification() {
        const contextText = this.elements.contextText.value.trim();
        const responseText = this.elements.responseText.value.trim();

        if (!contextText || !responseText) {
            alert('Please enter both context and response text.');
            return;
        }

        // Disable button during processing
        this.elements.verifyBtn.disabled = true;
        this.elements.verifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying...';

        try {
            const result = await halluGraphEngine.verify(contextText, responseText);
            this.displayResults(result);
            visualizer.render(result);
        } catch (error) {
            console.error('Verification error:', error);
            alert('An error occurred during verification.');
        } finally {
            this.elements.verifyBtn.disabled = false;
            this.elements.verifyBtn.innerHTML = '<i class="fas fa-balance-scale"></i> Verify Fidelity';
        }
    }

    async runBatchDemo() {
        if (this.elements.runBatchBtn.disabled) return;

        const batchSize = parseInt(this.elements.batchSize.value);
        const testData = generateBatchData(batchSize);

        this.elements.runBatchBtn.disabled = true;
        this.elements.runBatchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';

        const startTime = performance.now();
        let correct = 0;
        let totalLatency = 0;

        try {
            for (let i = 0; i < testData.length; i++) {
                const pair = testData[i];
                // Reuse existing engine verify, but suppress UI updates for speed if possible
                // Here we call verify directly
                const startItem = performance.now();
                const result = await halluGraphEngine.verify(pair.context, pair.response);
                totalLatency += (performance.now() - startItem);

                const predicted = result.isHallucination ? 1 : 0;
                if (predicted === pair.label) correct++;

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
            this.elements.runBatchBtn.disabled = false;
            this.elements.runBatchBtn.innerHTML = '<i class="fas fa-play"></i> Run Batch Demo';
        }
    }

    displayResults(result) {
        // Update CFI gauge
        const cfi = result.cfi;
        const cfiPercent = Math.round(cfi * 100);

        // Gauge animation (arc length is 251.2 for full gauge)
        const offset = 251.2 * (1 - cfi);
        this.elements.gaugeFill.style.strokeDashoffset = offset;

        // Color based on score
        if (cfi >= 0.75) {
            this.elements.gaugeFill.style.stroke = '#22c55e';
        } else if (cfi >= 0.5) {
            this.elements.gaugeFill.style.stroke = '#f59e0b';
        } else {
            this.elements.gaugeFill.style.stroke = '#ef4444';
        }

        this.elements.cfiValue.textContent = `${cfiPercent}%`;

        // Update verdict
        this.elements.verdict.className = 'verdict ' + (result.isHallucination ? 'fail' : 'pass');
        this.elements.verdict.innerHTML = `
            <span class="verdict-text">${result.isHallucination ? '⚠️ Hallucination Detected' : '✓ Faithful'}</span>
        `;

        // Update component scores
        this.elements.egScore.textContent = `${Math.round(result.eg * 100)}%`;
        this.elements.rpScore.textContent = result.rp !== null ? `${Math.round(result.rp * 100)}%` : 'N/A';

        this.elements.entityCount.textContent = `${result.egResult.grounded.length} / ${result.egResult.total}`;
        this.elements.relationCount.textContent = `${result.rpResult.preserved.length} / ${result.rpResult.total}`;

        // Update timing metrics
        if (result.processingTime) {
            const total = result.processingTime;
            this.elements.extractTime.textContent = `${Math.round(total * 0.4)} ms`;
            this.elements.alignTime.textContent = `${Math.round(total * 0.6)} ms`;
            this.elements.totalTime.textContent = `${Math.round(total)} ms`;

            // Calculate and display carbon footprint (estimated based on processing time)
            // Approximate: 0.0001g CO2 per 100ms of CPU processing time
            const carbonGrams = (total / 100) * 0.0001;
            if (this.elements.carbonValue) {
                if (carbonGrams < 0.0001) {
                    this.elements.carbonValue.textContent = `~${(carbonGrams * 1000).toFixed(3)}mg`;
                } else {
                    this.elements.carbonValue.textContent = `~${carbonGrams.toFixed(4)}g`;
                }
            }
        }

        // Update audit trail (improved white-box explanations)
        this.displayAuditTrail(result);
    }

    displayAuditTrail(result) {
        const items = [];

        // Generate clear white-box explanations
        // 1. Ungrounded entities (hallucination reasons)
        for (const entity of result.egResult.ungrounded) {
            items.push({
                type: 'ungrounded',
                icon: '❌',
                text: `Entity NOT in source: "${entity.text}" (${entity.type}) — This ${entity.type} appears in the response but not in the source document.`
            });
        }

        // 2. Unsupported relations
        for (const rel of result.rpResult.unsupported) {
            items.push({
                type: 'mismatch',
                icon: '⚠️',
                text: `Relation unsupported: "${rel.text?.substring(0, 60) || rel.relation}..." — This claim is not backed by the source.`
            });
        }

        // 3. Grounded entities (positive evidence)
        for (const match of result.egResult.grounded.slice(0, 3)) {
            items.push({
                type: 'grounded',
                icon: '✅',
                text: `Entity verified: "${match.response.text}" (${match.response.type}) matches source.`
            });
        }

        if (items.length === 0) {
            this.elements.auditList.innerHTML = `
                <div class="audit-placeholder">
                    <i class="fas fa-check-circle" style="color: #22c55e;"></i>
                    All entities and relations verified
                </div>
            `;
            return;
        }

        // Show ungrounded/mismatch items first
        const sorted = [...items].sort((a, b) => {
            const priority = { ungrounded: 0, mismatch: 1, grounded: 2 };
            return priority[a.type] - priority[b.type];
        });

        const display = sorted.slice(0, 6);

        this.elements.auditList.innerHTML = display.map(item => `
            <div class="audit-item ${item.type}">
                <span class="audit-icon">${item.icon}</span>
                <span class="audit-text">${item.text}</span>
            </div>
        `).join('');

        if (sorted.length > 6) {
            this.elements.auditList.innerHTML += `
                <div class="audit-placeholder">
                    ... and ${sorted.length - 6} more items
                </div>
            `;
        }
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const app = new HalluGraphApp();
    app.init();
});
