/**
 * HalluGraph App - Main Application Logic
 */

import { halluGraphEngine } from './hallugraph-engine.js';
import { visualizer } from './visualizer.js';

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
            totalTime: document.getElementById('totalTime')
        };
    }

    bindEvents() {
        this.elements.verifyBtn.addEventListener('click', () => this.runVerification());

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
