/**
 * Graph Visualizer - D3.js Force-Directed Layout
 * Renders alignment between source and hypothesis nodes
 */

class GraphVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = null;
        this.width = 0;
        this.height = 0;
        this.simulation = null;
    }

    /**
     * Initialize the SVG canvas
     */
    initialize() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        const svg = d3.select(`#${this.containerId} svg`);
        const rect = container.getBoundingClientRect();

        this.width = rect.width;
        this.height = rect.height;

        svg.attr('width', this.width)
            .attr('height', this.height);

        this.svg = svg;

        // Add gradient definitions
        const defs = svg.append('defs');

        // Source node gradient
        const sourceGradient = defs.append('radialGradient')
            .attr('id', 'sourceNodeGradient');
        sourceGradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#818cf8');
        sourceGradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#6366f1');

        // Hypothesis node gradient
        const hypGradient = defs.append('radialGradient')
            .attr('id', 'hypNodeGradient');
        hypGradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#a78bfa');
        hypGradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#8b5cf6');

        // Gauge gradient
        const gaugeGradient = defs.append('linearGradient')
            .attr('id', 'gaugeGradient')
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '0%');
        gaugeGradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#22c55e');
        gaugeGradient.append('stop')
            .attr('offset', '50%')
            .attr('stop-color', '#f59e0b');
        gaugeGradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#ef4444');

        // Create layer groups
        svg.append('g').attr('class', 'edges-layer');
        svg.append('g').attr('class', 'nodes-layer');
        svg.append('g').attr('class', 'labels-layer');

        this.showPlaceholder();
    }

    /**
     * Show placeholder when no data
     */
    showPlaceholder() {
        if (!this.svg) return;

        this.clear();

        this.svg.select('.labels-layer')
            .append('text')
            .attr('class', 'placeholder')
            .attr('x', this.width / 2)
            .attr('y', this.height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6b6b80')
            .attr('font-size', '14px')
            .text('Enter text and click Verify to see graph alignment');
    }

    /**
     * Clear the visualization
     */
    clear() {
        if (!this.svg) return;

        if (this.simulation) {
            this.simulation.stop();
        }

        this.svg.select('.edges-layer').selectAll('*').remove();
        this.svg.select('.nodes-layer').selectAll('*').remove();
        this.svg.select('.labels-layer').selectAll('*').remove();
    }

    /**
     * Render the alignment graph
     * @param {Object} data - Verification result details
     */
    render(data) {
        if (!this.svg || !data) {
            this.showPlaceholder();
            return;
        }

        this.clear();

        const { sourceSentences, hypSentences, alignmentMatrix } = data;

        // Create nodes
        const nodes = [];
        const nodeRadius = Math.min(20, Math.max(10, 300 / (sourceSentences.length + hypSentences.length)));

        // Source nodes (left side)
        sourceSentences.forEach((sentence, i) => {
            nodes.push({
                id: `src-${i}`,
                type: 'source',
                index: i,
                label: this.truncate(sentence, 30),
                fullText: sentence,
                x: this.width * 0.25,
                y: (i + 1) * this.height / (sourceSentences.length + 1)
            });
        });

        // Hypothesis nodes (right side)
        hypSentences.forEach((sentence, i) => {
            nodes.push({
                id: `hyp-${i}`,
                type: 'hypothesis',
                index: i,
                label: this.truncate(sentence, 30),
                fullText: sentence,
                x: this.width * 0.75,
                y: (i + 1) * this.height / (hypSentences.length + 1)
            });
        });

        // Create edges from alignment matrix
        const edges = [];
        const threshold = 0.1; // Only show significant alignments

        for (let i = 0; i < hypSentences.length; i++) {
            for (let j = 0; j < sourceSentences.length; j++) {
                const weight = alignmentMatrix[i][j];
                if (weight > threshold) {
                    edges.push({
                        source: `hyp-${i}`,
                        target: `src-${j}`,
                        weight: weight
                    });
                }
            }
        }

        // Set up force simulation
        this.simulation = d3.forceSimulation(nodes)
            .force('charge', d3.forceManyBody().strength(-100))
            .force('x', d3.forceX(d => d.type === 'source' ? this.width * 0.25 : this.width * 0.75).strength(0.8))
            .force('y', d3.forceY(d => d.y).strength(0.5))
            .force('collision', d3.forceCollide().radius(nodeRadius + 5));

        // Render edges
        const edgeSelection = this.svg.select('.edges-layer')
            .selectAll('line')
            .data(edges)
            .enter()
            .append('line')
            .attr('stroke', 'rgba(255, 255, 255, 0.3)')
            .attr('stroke-width', d => Math.max(1, d.weight * 5))
            .attr('stroke-opacity', d => 0.2 + d.weight * 0.6);

        // Render nodes
        const nodeSelection = this.svg.select('.nodes-layer')
            .selectAll('circle')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('r', nodeRadius)
            .attr('fill', d => d.type === 'source' ? 'url(#sourceNodeGradient)' : 'url(#hypNodeGradient)')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('stroke-opacity', 0.3)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) this.simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) this.simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }));

        // Add tooltips
        nodeSelection.append('title')
            .text(d => d.fullText);

        // Render labels
        const labelSelection = this.svg.select('.labels-layer')
            .selectAll('text')
            .data(nodes)
            .enter()
            .append('text')
            .attr('text-anchor', d => d.type === 'source' ? 'end' : 'start')
            .attr('dx', d => d.type === 'source' ? -nodeRadius - 5 : nodeRadius + 5)
            .attr('dy', 4)
            .attr('fill', '#a0a0b0')
            .attr('font-size', '10px')
            .text(d => d.label);

        // Node map for edge references
        const nodeMap = new Map(nodes.map(n => [n.id, n]));

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            edgeSelection
                .attr('x1', d => nodeMap.get(d.source)?.x || 0)
                .attr('y1', d => nodeMap.get(d.source)?.y || 0)
                .attr('x2', d => nodeMap.get(d.target)?.x || 0)
                .attr('y2', d => nodeMap.get(d.target)?.y || 0);

            nodeSelection
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            labelSelection
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        // Run simulation
        this.simulation.alpha(1).restart();
    }

    /**
     * Truncate text for labels
     */
    truncate(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }

    /**
     * Handle window resize
     */
    resize() {
        const container = document.getElementById(this.containerId);
        if (!container || !this.svg) return;

        const rect = container.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        this.svg.attr('width', this.width)
            .attr('height', this.height);
    }
}

// Export
export const graphVisualizer = new GraphVisualizer('graphContainer');
export default graphVisualizer;
