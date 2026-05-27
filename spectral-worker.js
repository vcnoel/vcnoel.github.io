import { AutoTokenizer, AutoModelForCausalLM, env } from
    'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0';

env.useBrowserCache = true;

const MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct';
const DTYPE = 'q4';

let tokenizer, model;
const history = [];

function dirichlet(A, n) {
    let E = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            const a = A[i * n + j];
            if (a < 1e-12) continue;
            const d = (i - j) / n;
            E += a * d * d;
        }
    return E / 2;
}

function entropy(A, n) {
    let H = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            const p = A[i * n + j];
            if (p > 1e-12) H -= p * Math.log2(p);
        }
    return H / n;
}

function localityScore(A, n) {
    let s = 0;
    for (let i = 0; i < n; i++) {
        let mx = 0;
        for (let j = 0; j < n; j++) mx = Math.max(mx, A[i * n + j]);
        s += mx;
    }
    return s / n;
}

function proxyMetrics(text) {
    const words = text.trim().split(/\s+/);
    const div = new Set(words).size / Math.max(words.length, 1);
    const n = 28;
    const dV = [], eV = [], lV = [];
    for (let l = 0; l < n; l++) {
        const t = l / (n - 1);
        dV.push(Math.max(0, 0.26 * (1 - t * 0.55) + (1 - div) * 0.04 + (Math.random() - 0.5) * 0.025));
        eV.push(Math.max(0, 3.1 * (1 - t * 0.38) + (1 - div) * 0.4 + (Math.random() - 0.5) * 0.15));
        lV.push(Math.max(0, Math.min(1, 0.34 + t * 0.38 + (Math.random() - 0.5) * 0.04)));
    }
    return { dV, eV, lV, real: false };
}

async function extractMetrics(inputIds, attnMask) {
    try {
        const out = await model(inputIds, { attention_mask: attnMask, output_attentions: true });
        if (!out.attentions || !out.attentions.length) return null;
        const seqLen = inputIds.dims[1];
        const dV = [], eV = [], lV = [];
        for (const layerAttn of out.attentions) {
            const heads = layerAttn.dims[1];
            const avg = new Float32Array(seqLen * seqLen);
            for (let h = 0; h < heads; h++) {
                const off = h * seqLen * seqLen;
                for (let i = 0; i < seqLen * seqLen; i++)
                    avg[i] += layerAttn.data[off + i] / heads;
            }
            dV.push(dirichlet(avg, seqLen));
            eV.push(entropy(avg, seqLen));
            lV.push(localityScore(avg, seqLen));
        }
        return { dV, eV, lV, real: true };
    } catch {
        return null;
    }
}

self.addEventListener('message', async ({ data }) => {
    const { type, text } = data;

    if (type === 'load') {
        try {
            const cb = p => self.postMessage({ type: 'progress', ...p });
            tokenizer = await AutoTokenizer.from_pretrained(MODEL, { progress_callback: cb });
            model = await AutoModelForCausalLM.from_pretrained(MODEL, { dtype: DTYPE, progress_callback: cb });
            self.postMessage({ type: 'ready' });
        } catch (e) {
            self.postMessage({ type: 'error', message: e.message });
        }
    }

    if (type === 'generate') {
        try {
            history.push({ role: 'user', content: text });

            const inputs = tokenizer.apply_chat_template(history, {
                add_generation_prompt: true,
                return_dict: true,
            });

            const inputLength = inputs.input_ids.dims[1];

            const { sequences } = await model.generate({
                ...inputs,
                max_new_tokens: 220,
                do_sample: true,
                temperature: 0.7,
                top_p: 0.9,
                repetition_penalty: 1.1,
                return_dict_in_generate: true,
            });

            const newTokenIds = Array.from(sequences.data).slice(inputLength);
            const reply = tokenizer.decode(newTokenIds, { skip_special_tokens: true });
            history.push({ role: 'assistant', content: reply });

            const real = await extractMetrics(inputs.input_ids, inputs.attention_mask);
            self.postMessage({ type: 'reply', text: reply, metrics: real ?? proxyMetrics(reply) });
        } catch (e) {
            self.postMessage({ type: 'error', message: e.message });
        }
    }
});
