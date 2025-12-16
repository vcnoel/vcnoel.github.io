/**
 * HalluGraph Engine - Knowledge Graph Alignment for Legal RAG
 * 
 * Based on: "HalluGraph: Auditable Hallucination Detection for Legal RAG Systems"
 * 
 * Metrics:
 * - Entity Grounding (EG): fraction of response entities in source
 * - Relation Preservation (RP): fraction of response edges supported by source
 * - Composite Fidelity Index (CFI): α·EG + (1-α)·RP
 */

class HalluGraphEngine {
    constructor() {
        this.params = {
            alpha: 0.7,         // Weight for EG in CFI
            threshold: 0.75,    // CFI threshold for pass/fail
            similarityThreshold: 0.7  // Embedding similarity threshold for matching
        };

        this.entityTypes = {
            PERSON: 'person',
            ORG: 'organization',
            DATE: 'date',
            MONEY: 'money',
            PERCENT: 'percent',
            CARDINAL: 'cardinal',
            PROVISION: 'provision'
        };
    }

    // =========================================================================
    // ENTITY EXTRACTION (using regex patterns for legal entities)
    // =========================================================================

    extractEntities(text) {
        const entities = [];

        // Money amounts (e.g., $45,000, $135,000.00)
        const moneyRegex = /\$[\d,]+(?:\.\d{2})?/g;
        let match;
        while ((match = moneyRegex.exec(text)) !== null) {
            entities.push({
                text: match[0],
                type: this.entityTypes.MONEY,
                start: match.index,
                normalized: parseFloat(match[0].replace(/[$,]/g, ''))
            });
        }

        // Dates (e.g., January 15, 2024, 01/15/2024)
        const dateRegex = /(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}\/\d{1,2}\/\d{4}|\d{4}/g;
        while ((match = dateRegex.exec(text)) !== null) {
            entities.push({
                text: match[0],
                type: this.entityTypes.DATE,
                start: match.index,
                normalized: match[0].toLowerCase().replace(/,/g, '')
            });
        }

        // Organizations (e.g., "XXX LLC", "XXX Inc.", "XXX Corp")
        const orgRegex = /([A-Z][a-zA-Z\s]+(?:LLC|Inc\.|Corp\.?|Corporation|Company|Properties|Group|Partners|LLP))/g;
        while ((match = orgRegex.exec(text)) !== null) {
            entities.push({
                text: match[0].trim(),
                type: this.entityTypes.ORG,
                start: match.index,
                normalized: match[0].trim().toLowerCase()
            });
        }

        // Legal roles (Landlord, Tenant, Plaintiff, Defendant)
        const roleRegex = /"(Landlord|Tenant|Plaintiff|Defendant|Petitioner|Respondent)"/g;
        while ((match = roleRegex.exec(text)) !== null) {
            entities.push({
                text: match[1],
                type: this.entityTypes.PERSON,
                start: match.index,
                normalized: match[1].toLowerCase()
            });
        }

        // Durations/years (e.g., "5 years", "3-year")
        const durationRegex = /\d+[ -]?years?/gi;
        while ((match = durationRegex.exec(text)) !== null) {
            entities.push({
                text: match[0],
                type: this.entityTypes.CARDINAL,
                start: match.index,
                normalized: parseInt(match[0])
            });
        }

        // Case citations (e.g., "Smith v. Jones", "500 U.S. 123")
        const caseRegex = /([A-Z][a-z]+)\s+v\.?\s+([A-Z][a-z]+)/g;
        while ((match = caseRegex.exec(text)) !== null) {
            entities.push({
                text: match[0],
                type: 'case',
                start: match.index,
                normalized: match[0].toLowerCase()
            });
        }

        const reporterRegex = /\d+\s+(?:U\.S\.|F\.\d+d?|S\.\s?Ct\.)\s+\d+/g;
        while ((match = reporterRegex.exec(text)) !== null) {
            entities.push({
                text: match[0],
                type: 'citation',
                start: match.index,
                normalized: match[0]
            });
        }

        // Provisions (e.g., "Section 4.2", "Article III")
        const provisionRegex = /(?:Section|Article|Clause|Paragraph)\s+[\dIVXLCDM]+(?:\.\d+)*/gi;
        while ((match = provisionRegex.exec(text)) !== null) {
            entities.push({
                text: match[0],
                type: this.entityTypes.PROVISION,
                start: match.index,
                normalized: match[0].toLowerCase()
            });
        }

        return entities;
    }

    // =========================================================================
    // RELATION EXTRACTION (simplified OpenIE-style)
    // =========================================================================

    extractRelations(text, entities) {
        const relations = [];

        // Common legal relation patterns
        const patterns = [
            { regex: /(\w+(?:\s+\w+)*)\s+(?:entered into|signed|executed)\s+(?:a|an)\s+(\w+(?:\s+\w+)*)/gi, type: 'signed' },
            { regex: /([\w\s]+)\s+(?:is|was)\s+(?:payable|due)\s+(?:on|by)\s+([\w\s,]+)/gi, type: 'payable' },
            { regex: /(?:monthly\s+)?rent\s+(?:is|of)\s+(\$[\d,]+)/gi, type: 'rent_amount' },
            { regex: /lease\s+(?:term|runs)\s+(?:is|for)?\s*(\d+\s+years?)/gi, type: 'term' },
            { regex: /([\w\s]+)\s+held\s+that\s+([\w\s,]+)/gi, type: 'held' },
            { regex: /([\w\s]+)\s+filed\s+(?:a\s+)?(\w+)\s+against\s+([\w\s]+)/gi, type: 'filed' },
            { regex: /(?:Section|Article|Clause)\s+([\w\.]+)\s+(?:requires|states|mandates)\s+(?:that\s+)?([\w\s]+?)(?:\s+within|\.|$)/gi, type: 'requires' }
        ];

        for (const pattern of patterns) {
            let match;
            while ((match = pattern.regex.exec(text)) !== null) {
                relations.push({
                    subject: match[1]?.trim() || '',
                    relation: pattern.type,
                    object: match[2]?.trim() || '',
                    text: match[0]
                });
            }
        }

        // Also extract entity-to-entity relations based on proximity
        for (let i = 0; i < entities.length - 1; i++) {
            const e1 = entities[i];
            const e2 = entities[i + 1];
            const gap = e2.start - (e1.start + e1.text.length);

            if (gap > 0 && gap < 50) {
                relations.push({
                    subject: e1.text,
                    relation: 'adjacent',
                    object: e2.text,
                    subjectType: e1.type,
                    objectType: e2.type
                });
            }
        }

        return relations;
    }

    // =========================================================================
    // ENTITY GROUNDING (EG)
    // Fraction of response entities that appear in source
    // =========================================================================

    computeEntityGrounding(sourceEntities, responseEntities) {
        const groundedEntities = [];
        const ungroundedEntities = [];

        for (const resEntity of responseEntities) {
            let grounded = false;
            let matchedSource = null;

            for (const srcEntity of sourceEntities) {
                // Type must match
                if (srcEntity.type !== resEntity.type) continue;

                // Check normalized values
                if (srcEntity.type === this.entityTypes.MONEY) {
                    if (srcEntity.normalized === resEntity.normalized) {
                        grounded = true;
                        matchedSource = srcEntity;
                        break;
                    }
                } else if (srcEntity.type === this.entityTypes.CARDINAL) {
                    if (srcEntity.normalized === resEntity.normalized) {
                        grounded = true;
                        matchedSource = srcEntity;
                        break;
                    }
                } else {
                    // String comparison with normalization
                    if (srcEntity.normalized === resEntity.normalized ||
                        srcEntity.normalized.includes(resEntity.normalized) ||
                        resEntity.normalized.includes(srcEntity.normalized)) {
                        grounded = true;
                        matchedSource = srcEntity;
                        break;
                    }
                }
            }

            if (grounded) {
                groundedEntities.push({ response: resEntity, source: matchedSource });
            } else {
                ungroundedEntities.push(resEntity);
            }
        }

        const eg = responseEntities.length > 0
            ? groundedEntities.length / responseEntities.length
            : 1;

        return {
            score: eg,
            grounded: groundedEntities,
            ungrounded: ungroundedEntities,
            total: responseEntities.length
        };
    }

    // =========================================================================
    // RELATION PRESERVATION (RP)
    // Fraction of response relations supported by source
    // =========================================================================

    computeRelationPreservation(sourceRelations, responseRelations) {
        const preservedRelations = [];
        const unsupportedRelations = [];

        for (const resRel of responseRelations) {
            let preserved = false;

            for (const srcRel of sourceRelations) {
                // Same relation type and similar content
                if (srcRel.relation === resRel.relation) {
                    // Check if endpoints are similar
                    const subjectMatch = this.fuzzyMatch(srcRel.subject, resRel.subject);
                    const objectMatch = this.fuzzyMatch(srcRel.object, resRel.object);

                    if (subjectMatch > 0.5 || objectMatch > 0.5) {
                        preserved = true;
                        preservedRelations.push({ response: resRel, source: srcRel });
                        break;
                    }
                }
            }

            if (!preserved) {
                unsupportedRelations.push(resRel);
            }
        }

        // Edge-aware policy: if no relations, exclude from aggregation
        const rp = responseRelations.length > 0
            ? preservedRelations.length / responseRelations.length
            : null;

        return {
            score: rp,
            preserved: preservedRelations,
            unsupported: unsupportedRelations,
            total: responseRelations.length
        };
    }

    fuzzyMatch(str1, str2) {
        if (!str1 || !str2) return 0;
        const s1 = str1.toLowerCase().replace(/[\$,]/g, '').trim();
        const s2 = str2.toLowerCase().replace(/[\$,]/g, '').trim();

        // Strict check for numbers/dates (if they contain digits)
        const hasDigits = /\d/.test(s1) && /\d/.test(s2);
        if (hasDigits) {
            // For numbers/dates, we require exact match or very high similarity
            return s1 === s2 ? 1 : 0;
        }

        if (s1 === s2) return 1;
        if (s1.includes(s2) || s2.includes(s1)) return 0.8;

        // Jaccard similarity on words
        const words1 = new Set(s1.split(/\s+/));
        const words2 = new Set(s2.split(/\s+/));
        const intersection = [...words1].filter(w => words2.has(w)).length;
        const union = new Set([...words1, ...words2]).size;
        return union > 0 ? intersection / union : 0;
    }

    // =========================================================================
    // COMPOSITE FIDELITY INDEX (CFI)
    // =========================================================================

    computeCFI(eg, rp) {
        // If RP is null (no relations), rely solely on EG
        if (rp === null) {
            return eg;
        }
        return this.params.alpha * eg + (1 - this.params.alpha) * rp;
    }

    // =========================================================================
    // AUDIT TRAIL GENERATION
    // =========================================================================

    generateAuditTrail(egResult, rpResult) {
        const auditItems = [];

        // Grounded entities
        for (const item of egResult.grounded) {
            auditItems.push({
                type: 'grounded',
                icon: '✓',
                text: `Entity grounded: "${item.response.text}" (${item.response.type}) matches source`
            });
        }

        // Ungrounded entities
        for (const entity of egResult.ungrounded) {
            auditItems.push({
                type: 'ungrounded',
                icon: '✗',
                text: `⚠️ Ungrounded: "${entity.text}" (${entity.type}) not found in source`
            });
        }

        // Preserved relations
        for (const item of rpResult.preserved) {
            auditItems.push({
                type: 'grounded',
                icon: '✓',
                text: `Relation preserved: "${item.response.relation}"`
            });
        }

        // Unsupported relations
        for (const rel of rpResult.unsupported) {
            auditItems.push({
                type: 'mismatch',
                icon: '⚠',
                text: `Relation unsupported: "${rel.text?.substring(0, 50)}..."`
            });
        }

        return auditItems;
    }

    // =========================================================================
    // MAIN VERIFICATION
    // =========================================================================

    async verify(contextText, responseText) {
        const startTime = performance.now();

        // Extract entities from context and response
        const contextEntities = this.extractEntities(contextText);
        const responseEntities = this.extractEntities(responseText);

        // Extract relations
        const contextRelations = this.extractRelations(contextText, contextEntities);
        const responseRelations = this.extractRelations(responseText, responseEntities);

        // Compute Entity Grounding
        const egResult = this.computeEntityGrounding(contextEntities, responseEntities);

        // Compute Relation Preservation
        const rpResult = this.computeRelationPreservation(contextRelations, responseRelations);

        // Compute CFI
        const cfi = this.computeCFI(egResult.score, rpResult.score);

        // Generate audit trail
        const auditTrail = this.generateAuditTrail(egResult, rpResult);

        // Decision
        const isHallucination = cfi < this.params.threshold;

        return {
            cfi,
            eg: egResult.score,
            rp: rpResult.score,
            isHallucination,
            verdict: isHallucination ? 'FAIL' : 'PASS',
            egResult,
            rpResult,
            auditTrail,
            graphs: {
                context: {
                    entities: contextEntities,
                    relations: contextRelations
                },
                response: {
                    entities: responseEntities,
                    relations: responseRelations
                }
            },
            processingTime: performance.now() - startTime
        };
    }
}

export const halluGraphEngine = new HalluGraphEngine();
export default halluGraphEngine;
