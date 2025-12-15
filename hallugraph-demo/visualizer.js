/**
 * HalluGraph Visualizer - Knowledge Graph Rendering
 * 
 * Renders source and response knowledge graphs with alignment edges
 */

class HalluGraphVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.svg = d3.select('#graphSvg');
        this.width = 0;
        this.height = 0;
    }

    initialize() {
        this.updateDimensions();
        window.addEventListener('resize', () => this.updateDimensions());

        // Clear and set up SVG
        this.svg.selectAll('*').remove();

        // Add defs for gradients
        const defs = this.svg.append('defs');

        // Source entity gradient
        const sourceGrad = defs.append('linearGradient')
            .attr('id', 'sourceGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        sourceGrad.append('stop').attr('offset', '0%').attr('stop-color', '#059669');
        sourceGrad.append('stop').attr('offset', '100%').attr('stop-color', '#10b981');

        // Response entity gradient
        const responseGrad = defs.append('linearGradient')
            .attr('id', 'responseGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        responseGrad.append('stop').attr('offset', '0%').attr('stop-color', '#d4af37');
        responseGrad.append('stop').attr('offset', '100%').attr('stop-color', '#f4d789');

        // Ungrounded entity gradient
        const ungroundedGrad = defs.append('linearGradient')
            .attr('id', 'ungroundedGradient')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        ungroundedGrad.append('stop').attr('offset', '0%').attr('stop-color', '#ef4444');
        ungroundedGrad.append('stop').attr('offset', '100%').attr('stop-color', '#f87171');

        // Add groups for layering
        this.svg.append('g').attr('class', 'edges-layer');
        this.svg.append('g').attr('class', 'alignments-layer');
        this.svg.append('g').attr('class', 'nodes-layer');
        this.svg.append('g').attr('class', 'labels-layer');
    }

    updateDimensions() {
        if (this.container) {
            const rect = this.container.getBoundingClientRect();
            this.width = rect.width;
            this.height = rect.height;
            this.svg.attr('viewBox', `0 0 ${this.width} ${this.height}`);
        }
    }

    render(result) {
        if (!result || !result.graphs) {
            this.renderPlaceholder();
            return;
        }

        this.updateDimensions();

        const { context, response } = result.graphs;
        const { grounded, ungrounded } = result.egResult;

        // Clear previous content
        this.svg.select('.edges-layer').selectAll('*').remove();
        this.svg.select('.alignments-layer').selectAll('*').remove();
        this.svg.select('.nodes-layer').selectAll('*').remove();
        this.svg.select('.labels-layer').selectAll('*').remove();

        // Position source entities on left, response entities on right
        const sourceNodes = this.positionNodes(context.entities, 'source', 0.2);
        const responseNodes = this.positionNodes(response.entities, 'response', 0.8);

        // Mark grounded/ungrounded
        const ungroundedTexts = new Set(ungrounded.map(e => e.text));
        for (const node of responseNodes) {
            node.grounded = !ungroundedTexts.has(node.entity.text);
        }

        // Draw alignment edges for grounded entities
        this.drawAlignments(sourceNodes, responseNodes, grounded);

        // Draw entity nodes
        this.drawNodes(sourceNodes, 'source');
        this.drawNodes(responseNodes, 'response');

        // Draw labels
        this.drawLabels([...sourceNodes, ...responseNodes]);
    }

    positionNodes(entities, side, xFraction) {
        const nodes = [];
        const padding = 50;
        const usableHeight = this.height - 2 * padding;
        const x = this.width * xFraction;

        // Remove duplicates
        const uniqueEntities = [];
        const seen = new Set();
        for (const e of entities) {
            if (!seen.has(e.text)) {
                seen.add(e.text);
                uniqueEntities.push(e);
            }
        }

        const spacing = Math.min(50, usableHeight / (uniqueEntities.length || 1));
        const startY = padding + (usableHeight - spacing * (uniqueEntities.length - 1)) / 2;

        for (let i = 0; i < uniqueEntities.length; i++) {
            nodes.push({
                id: `${side}-${i}`,
                entity: uniqueEntities[i],
                x: x + (Math.random() - 0.5) * 30,
                y: startY + i * spacing,
                side
            });
        }

        return nodes;
    }

    drawAlignments(sourceNodes, responseNodes, grounded) {
        const alignmentsLayer = this.svg.select('.alignments-layer');

        for (const match of grounded) {
            const sourceNode = sourceNodes.find(n =>
                n.entity.text === match.source.text ||
                n.entity.normalized === match.source.normalized
            );
            const responseNode = responseNodes.find(n =>
                n.entity.text === match.response.text
            );

            if (sourceNode && responseNode) {
                alignmentsLayer.append('line')
                    .attr('x1', sourceNode.x)
                    .attr('y1', sourceNode.y)
                    .attr('x2', responseNode.x)
                    .attr('y2', responseNode.y)
                    .attr('stroke', '#22c55e')
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.4)
                    .attr('stroke-dasharray', '5,5');
            }
        }
    }

    drawNodes(nodes, side) {
        const nodesLayer = this.svg.select('.nodes-layer');

        for (const node of nodes) {
            let fill;
            if (side === 'source') {
                fill = 'url(#sourceGradient)';
            } else {
                fill = node.grounded ? 'url(#responseGradient)' : 'url(#ungroundedGradient)';
            }

            nodesLayer.append('circle')
                .attr('cx', node.x)
                .attr('cy', node.y)
                .attr('r', 20)
                .attr('fill', fill)
                .attr('stroke', side === 'source' ? '#059669' : (node.grounded ? '#d4af37' : '#ef4444'))
                .attr('stroke-width', 2)
                .attr('opacity', 0.9)
                .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))');
        }
    }

    drawLabels(nodes) {
        const labelsLayer = this.svg.select('.labels-layer');

        for (const node of nodes) {
            // Truncate long labels
            const label = node.entity.text.length > 15
                ? node.entity.text.substring(0, 12) + '...'
                : node.entity.text;

            // Position label to the side
            const labelX = node.side === 'source'
                ? node.x - 28
                : node.x + 28;
            const anchor = node.side === 'source' ? 'end' : 'start';

            labelsLayer.append('text')
                .attr('x', labelX)
                .attr('y', node.y)
                .attr('dy', '0.35em')
                .attr('text-anchor', anchor)
                .attr('fill', '#a0a0b0')
                .attr('font-size', '10px')
                .attr('font-family', 'JetBrains Mono, monospace')
                .text(label);
        }
    }

    renderPlaceholder() {
        this.svg.select('.nodes-layer').selectAll('*').remove();
        this.svg.select('.labels-layer').selectAll('*').remove();
        this.svg.select('.alignments-layer').selectAll('*').remove();

        this.svg.select('.labels-layer')
            .append('text')
            .attr('x', '50%')
            .attr('y', '50%')
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b6b80')
            .attr('font-size', '14px')
            .text('Knowledge graphs will appear here');
    }
}

export const visualizer = new HalluGraphVisualizer('graphContainer');
export default visualizer;
